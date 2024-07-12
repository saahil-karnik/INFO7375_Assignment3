# pinecone_setup.py
import os
import pinecone
import openai
from dotenv import load_dotenv
from mock_data import generate_mock_data

# Load environment variables
load_dotenv()

def initialize_pinecone():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is not set in environment variables")

    pinecone.init(api_key=pinecone_api_key)
    index_name = 'book-recommendations'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=512)  # Assuming using 512-dimensional vectors

    return pinecone.Index(index_name)

def generate_embeddings(text):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set in environment variables")

    openai.api_key = openai_api_key
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    if 'data' in response and response['data']:
        return response['data'][0]['embedding']
    else:
        raise ValueError("Invalid response from OpenAI API: No embedding found")

def add_data_to_pinecone():
    index = initialize_pinecone()
    mock_data = generate_mock_data()

    for i, book in enumerate(mock_data):
        embedding = generate_embeddings(book['summary'] + " " + " ".join(book['tags']))
        # Ensure the data structure matches Pinecone's expected format
        index.upsert([(str(i), embedding, {'metadata': book})])  # Adding metadata to the upsert

if __name__ == '__main__':
    add_data_to_pinecone()
