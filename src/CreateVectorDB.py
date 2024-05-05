from pinecone import Pinecone, PodSpec
from openai import OpenAI
import pandas as pd
import json
import streamlit as st

pc = Pinecone(api_key='50006779-0bb6-4f4d-b1f4-2b081ee4b5c9')
# api_key1 = 2cf8f781-fed8-4d04-9469-710b2d3dcb33
# api_key2 = 71c551fa-9d7e-4375-b3d4-1b4c876a9edd
pc.create_index(
  name="hackdavis",
  dimension=3072,
  metric="cosine",
  spec=PodSpec(
    environment="gcp-starter"
  )
)
index = pc.Index("hackdavis")
# Load your dataset

df = pd.read_csv("C:\\Users\\srini\\Downloads\\HealthCareBot\\trainset.csv")

texts = []
for i in range(0,3000):
    text = f"Instruction:{df['input'][i]}\nResponse:{df['output'][i]}"
    texts.append(text)

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
vectors_embedding = []


for text in texts:
    # Generate embedding for each description
    embedding_response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # Choose the appropriate embeddings model
    )
    
    # Extract the embedding vector
    vector = embedding_response.data[0].embedding
    vector_length = len(vector)
    # Prepare the vector for insertion, including metadata
    vectors_embedding.append({
        "id": str(i),
        "values": vector,
        "metadata": {"description": text }  # Add metadata here
    })

# Save the embeddings to a JSON file
with open('embeddings.json', 'w') as file:
    json.dump(vectors_embedding, file)
print("Vector Length", vector_length)
print("Embeddings saved to embeddings.json")

import json
import itertools
# Load the JSON file
with open('embeddings.json', 'r') as file:
    vector_embeddings = json.load(file)



def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))





# Upsert data with 100 vectors per upsert request
for ids_vectors_chunk in chunks(vector_embeddings, batch_size=100):
    index.upsert(vectors=ids_vectors_chunk)

