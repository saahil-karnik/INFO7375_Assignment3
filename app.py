import os
import streamlit as st
import pinecone
from gpt4all import GPT4All
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))
index = pinecone.Index('book-recommendations')

# Load GPT4All model
model_path = "C:/Users/33641/Downloads/gpt4all-lora-quantized.bin"  # Update this with your model file path
model = GPT4All(model_name=model_path)

# Function to generate embeddings
def generate_embeddings(model, text):
    response = model.generate(text)
    if 'data' in response and response['data']:
        return response['data'][0].get('embedding')
    else:
        raise ValueError("Invalid response from model: No embedding found.")

# Function to get book recommendations
def get_recommendations(query):
    embedding = generate_embeddings(model, query)
    result = index.query(embedding, top_k=3, include_metadata=True)
    return result['matches']

# Streamlit App
st.title('Personalized Book Recommendations')

user_input = st.text_input("Enter your book preferences:")
if user_input:
    try:
        recommendations = get_recommendations(user_input)
        for rec in recommendations:
            st.subheader(rec['metadata']['title'])
            st.write(f"Author: {rec['metadata']['author']}")
            st.write(f"Genre: {', '.join(rec['metadata']['genre'])}")
            st.write(f"Summary: {rec['metadata']['summary']}")
            st.write(f"User Ratings: {rec['metadata']['user_ratings']}")
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")