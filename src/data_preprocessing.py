import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.faiss_index import FAISSIndex
from utils.database import get_db_connection

load_dotenv()

def preprocess_data():
    # Load data
    data = pd.read_csv("data/raw/soil_data.csv")
    data["clean_text"] = data["raw_text"].apply(lambda x: x.lower().strip())
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data["clean_text"].tolist())
    
    # Save processed data
    data.to_parquet("data/processed/processed_data.parquet")
    np.save("data/processed/embeddings.npy", embeddings)
    
    # Store in PostgreSQL + FAISS
    conn = get_db_connection()
    cur = conn.cursor()
    faiss_index = FAISSIndex()
    
    for idx, (text, emb) in enumerate(zip(data["clean_text"], embeddings)):
        cur.execute(
            "INSERT INTO documents (content, embedding_id) VALUES (%s, %s)",
            (text, idx)
        )
        faiss_index.add(emb, idx)
    
    conn.commit()
    cur.close()
    conn.close()
    faiss_index.save("models/faiss_index.bin")

if __name__ == "__main__":
    preprocess_data()