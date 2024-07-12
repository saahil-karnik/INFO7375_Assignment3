import os
from gpt4all import GPT4All
from mock_data import generate_mock_data
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_pinecone():
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))
    index_name = 'book-recommendations'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=512)  # Assuming using 512-dimensional vectors

    return pinecone.Index(index_name)

def generate_embeddings(model, text):
    response = model.generate(text)
    # Check if the response structure contains 'data' and 'embedding'
    if 'data' in response and response['data']:
        return response['data'][0].get('embedding')
    else:
        raise ValueError("Invalid response from model: No embedding found.")

def add_data_to_pinecone(model):
    index = initialize_pinecone()
    mock_data = generate_mock_data()

    for i, book in enumerate(mock_data):
        embedding = generate_embeddings(model, book['summary'] + " " + " ".join(book['tags']))
        # Ensure the data structure matches Pinecone's expected format
        index.upsert([(str(i), embedding, {'metadata': book})])  # Adding metadata to the upsert

if __name__ == '__main__':
    model_path = "C:/Users/33641/Downloads/gpt4all-lora-quantized.bin"  # Update this with your model file path
    model = GPT4All(model_name=model_path)
    add_data_to_pinecone(model)
