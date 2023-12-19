import datetime
import json
import os
import random
import tempfile
import openai
import pinecone
from dotenv import load_dotenv
from flask import jsonify, Flask, request
from flask_cors import CORS
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import XataChatMessageHistory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate
from xata import XataClient

app = Flask(__name__)
CORS(app)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_ENV"] = os.getenv('PINECONE_ENV')

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)

index_name = os.getenv('PINECONE_INDEX_NAME')
documents = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
retrieverdb = documents.as_retriever()
retrieverdb.search_type = "similarity"

xata_api_key = os.getenv("XATA_API_KEY")
db_url = os.getenv('XATA_URI')
xata = XataClient(
    db_url=db_url,
    api_key=xata_api_key,
)
_DEFAULT_SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

template = """
Given are the following context (delimited by <ctx></ctx>) and the chat histotry (delimited by <hs></hs>).
if the question is a gretting or nice words, answer nicely.Say that you don't know if the question is about something not mentioned in the context.
Always rephrase the question only based on the context   
and answer the follow up question with the context.
-----
<ctx>
{context}
</ctx>
------
<hs>
{chat_history}
</hs>   
-----
{question}  
"""
prompt = PromptTemplate(
    input_variables=['chat_history', 'context', 'question'],
    template=template,
)

index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def del_exec(id_vector, unique_id):
    not_found = False
    ids = []
    last_id = id_vector
    m_unique_id = unique_id
    data = index.query(id=id_vector, filter={'unique id': unique_id},
                       top_k=100, include_metadata=True)
    if 'matches' in data and isinstance(data['matches'], list):
        matches = data['matches']
        for i in range(len(matches)):
            if 'id' in matches[i]:
                if i < len(matches) - 1:
                    ids.append(matches[i]['id'])
                else:
                    last_id = matches[i]['id']
                    not_found = True

    for this_id in ids:
        index.delete(id=this_id, filter={'unique id': data['matches'][0]['metadata']['unique id']})
    if not not_found:
        del_exec(last_id, m_unique_id)
    else:
        index.delete(id=last_id, filter={'unique id': data['matches'][0]['metadata']['unique id']})
        return


@app.post('/overwrite')
def delete_doc():
    try:
        authorization_header = request.headers.get('Authorization')
        if authorization_header is None:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        api_key = parts[1]
        if api_key == os.getenv('BACKEND_API_KEY'):
            pass
        else:
            return jsonify({"error": "Unauthorized source"}), 403

        metadata_str = request.form.get('allMetadata')
        metadata_ = json.loads(metadata_str)
        metadata = json.loads(metadata_)

        unique_id_value = next((item['value'] for item in metadata if item['key'] == 'unique id'), None)
        metric_dimension = 1536
        vector_for_query = [random.uniform(0.0, 1.0) for _ in range(metric_dimension)]
        result = index.query(vector=vector_for_query, top_k=1, filter={'unique id': unique_id_value})
        if len(result['matches']) != 0:
            del_exec(result['matches'][0]['id'], unique_id_value)
            if request.headers.get('Delete') is not None:
                file = request.files['file']
                process_files(file, metadata)
            else:
                return jsonify({'success': f'The file has been successfully deleted!'})
            return jsonify({'success': f'The file has been successfully overwritten!'})
        return jsonify({'error': f' No file found with the corresponded unique Id !'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def modify_metadata(id_vector, allmetadata, unique_ID):
    metric_dimension = 1536
    not_found = False
    ids = []
    last_id = id_vector
    m_unique_id = unique_ID
    data = index.query(id=id_vector, filter={'unique id': m_unique_id},
                       top_k=50,
                       include_metadata=True)
    if 'matches' in data and isinstance(data['matches'], list):
        matches = data['matches']
        for i in range(len(matches)):
            if 'id' in matches[i]:
                if i < len(matches) - 1:
                    ids.append(matches[i]['id'])
                else:
                    last_id = matches[i]['id']
                    not_found = True
    new_metadata = {}
    for this_id in ids:
        update_exec(allmetadata, this_id)
    if not not_found:
        modify_metadata(last_id, allmetadata, m_unique_id)
    else:
        update_exec(allmetadata, last_id)
    return True


def update_exec(allmetadata, last_id):
    for metadata in allmetadata:
        key = metadata.get('key')
        value = metadata.get('value')
        if key == 'departments':
            departs = value.split(',')
            new_metadata = {
                str(key): departs
            }
            index.update(id=last_id, set_metadata=new_metadata)
        elif key == 'unique id':
            pass
        else:
            new_metadata = {
                repr({key}): repr({value})
            }
            index.update(id=last_id, set_metadata=new_metadata)


@app.post('/update_metadata')
def search_same_doc():
    try:
        authorization_header = request.headers.get('Authorization')
        if authorization_header is None:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        api_key = parts[1]
        if api_key == os.getenv('BACKEND_API_KEY'):
            pass
        else:
            return jsonify({"error": "Unauthorized source"}), 403

        metric_dimension = 1536
        metadata_str = request.form.get('allMetadata')
        metadata_ = json.loads(metadata_str)
        metadata = json.loads(metadata_)
        unique_id_value = next((item['value'] for item in metadata if item['key'] == 'unique id'), None)

        vector_for_query = [random.uniform(0.0, 1.0) for _ in range(metric_dimension)]
        result = index.query(vector=vector_for_query, top_k=1, filter={'unique id': unique_id_value})
        data = result['matches']
        if len(data) != 0:
            modify_metadata(result['matches'][0]['id'], metadata, unique_id_value)
            return jsonify({"success": "The file has been succesfully updated"})
        else:
            return jsonify({"error": "A document with this unique ID has not been found"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def queryByID(unique_ID):
    metric_dimension = 1536
    vector_for_query = [random.uniform(0.0, 1.0) for _ in range(metric_dimension)]
    result = index.query(vector=vector_for_query, top_k=1, filter={'unique id': unique_ID})
    data = result['matches']
    if len(data) != 0:
        return True
    return False


@app.post('/upload')
def upload_file():
    try:
        authorization_header = request.headers.get('Authorization')
        if authorization_header is None:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        api_key = parts[1]
        if api_key == os.getenv('BACKEND_API_KEY'):
            pass
        else:
            return jsonify({"error": "Unauthorized source"}), 403
        file = request.files['file']
        filename = file.filename

        if filename == '':
            return jsonify({'error': 'No selected file'})
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type'})

        metadata_str = request.form.get('allMetadata')
        metadata_ = json.loads(metadata_str)
        metadata = json.loads(metadata_)

        unique_id_value = next((item['value'] for item in metadata if item['key'] == 'unique id'), None)

        if queryByID(unique_id_value):
            return jsonify({"error": "Unique ID already exist!"})
        else:
            process_files(file, metadata)

        return jsonify({'success': f'The file {filename} has been successfully stored!'})
    except Exception as e:
        return jsonify({'error': str(e)})


def process_files(files, allmetadata):
    try:
        processed_docs = []
        file_extension = files.filename.rsplit('.', 1)[1].lower()
        pdf_content = files.read()

        if file_extension == 'pdf':
            Loader = PyPDFLoader
        elif file_extension == 'txt':
            Loader = TextLoader
        elif file_extension == 'csv':
            Loader = CSVLoader
        else:
            return jsonify({'error': 'Invalid file type'})

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=450)

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(pdf_content)
            loader = Loader(temp_file.name)
            documents = loader.load()

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            docs = splitter.split_documents(documents)

            for i, doc in enumerate(docs):
                for metadata in allmetadata:
                    key = metadata.get('key')
                    value = metadata.get('value')
                    if key == 'departments':
                        departs = value.split(',')
                        doc.metadata[str(key)] = departs
                    else:
                        doc.metadata[str(key)] = value
                        doc.metadata['last modified: '] = current_datetime
                processed_docs.append(doc)

        store_to_index(processed_docs)
    except FileNotFoundError as e:
        return jsonify({'error': f'File not found: {e}'})
    except Exception as e:
        return jsonify({'error': str(e)})


def store_to_index(file):
    if not file:
        return None

    index_name = os.getenv('PINECONE_INDEX_NAME')
    docsearch = Pinecone.from_documents(file, embeddings, index_name=index_name)
    return docsearch


@app.post('/get-answer')
def get_answer():
    try:
        authorization_header = request.headers.get('Authorization')
        if authorization_header is None:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        api_key = parts[1]
        if api_key == os.getenv('BACKEND_API_KEY'):
            pass
        else:
            return jsonify({"error": "Unauthorized source"}), 403
        user_input = request.form.get('user_message')
        metadataFilter = {"departments": request.form.get('department')}
        retrieverdb = documents.as_retriever()
        retrieverdb.search_type = "similarity"
        retrieverdb.search_kwargs = {'k': 4, 'filter': metadataFilter}
        user_id = request.form.get('user_id')
        chatId = request.form.get('chatId')

        chat_memory = XataChatMessageHistory(
            db_url=db_url,
            api_key=xata_api_key,
            session_id=chatId,
            table_name=user_id,
        )
        # llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5,max_tokens = )
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key='chat_history',
            input_key='question',
            memory=chat_memory,
            verbose=True,
            max_token_limit=2000,
            return_messages=False,
            streaming=True,
            prompt=SUMMARY_PROMPT
        )

        if len(chat_memory.messages):
            if len(chat_memory.messages) == 15:
                for _ in range(len(chat_memory.messages) - 10):
                    chat_memory.messages.pop(0)
            for i in range(len(chat_memory.messages) - 1):
                memory.save_context(
                    {'question': chat_memory.messages[i].content},
                    {'answer': chat_memory.messages[i + 1].content},
                )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retrieverdb,
                chain_type='stuff',
                chain_type_kwargs={
                    'memory': memory,
                    'prompt': prompt,
                    'verbose': True
                }
            )
            response = chain({'query': user_input})
            chat_memory.add_user_message(user_input)
            chat_memory.add_ai_message(response['result'])
            return jsonify({"success": response['result']})

        else:
            chain = RetrievalQA.from_chain_type(
                verbose=True,
                llm=llm,
                retriever=retrieverdb,
                chain_type='stuff',
                chain_type_kwargs={
                    'memory': memory,
                    'prompt': prompt,
                    'verbose': True
                },
            )
            response = chain({'query': user_input})
            chat_memory.add_user_message(user_input)
            chat_memory.add_ai_message(response['result'])
            return jsonify({"success": response['result']})

    except Exception as e:
        if e == 'Failed to connect; did you specify the correct index name?':
            return jsonify({'error': 'Failed to connect with the server. Please retry!'})
        words = str(e).split()
        if words[0] == 'Error' and words[1] == 'running' and words[2] == 'query:':
            return jsonify({'error': 'Please provide an user id'})
        elif words[0] == 'Index':
            return jsonify({'error': 'There is no index corresponding to the provided index-name!'})

        return jsonify({"error": str(e)}), 500


@app.get('/home')
def hello():
    return 'Hello, here is the Flask backend service home page'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
