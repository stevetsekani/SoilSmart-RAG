import os
import requests
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.faiss_index import FAISSIndex
from utils.database import get_db_connection

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')

def perplexity_api_query(prompt: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    
    try:
        response = requests.post(
            url,
            json={
                "model": "sonar-medium-online",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()  
        
        # Debug: Print full API response
        print(f"\n[DEBUG] API Response: {response.text}")  # Add this line
        
        return response.json()["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] API Request Failed: {str(e)}")
        return ""
    except KeyError:
        print(f"\n[ERROR] Malformed API Response: {response.text}")
        return ""
def rag_query(query: str, top_k=3):
    print(f"\n[DEBUG] Starting RAG query: {query}")  # Debug 1
    
    # Retrieve documents
    query_embedding = model.encode([query])[0]
    print(f"[DEBUG] Query embedding shape: {query_embedding.shape}")  # Debug 2
    
    try:
        faiss_index = FAISSIndex.load("models/faiss_index.bin")
        print(f"[DEBUG] FAISS index loaded with {faiss_index.index.ntotal} vectors")  # Debug 3
    except Exception as e:
        print(f"[ERROR] FAISS load failed: {str(e)}")
        return ""

    doc_indices = faiss_index.search(query_embedding, top_k)
    print(f"[DEBUG] Retrieved indices: {doc_indices}")  # Debug 4
    
    # Fetch context
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        print("[DEBUG] Database connection established")  # Debug 5
    except Exception as e:
        print(f"[ERROR] Database connection failed: {str(e)}")
        return ""

    context = []
    for idx in doc_indices:
        try:
            cur.execute("SELECT content FROM documents WHERE embedding_id = %s", (int(idx),))
            result = cur.fetchone()
            if result:
                context.append(result[0])
                print(f"[DEBUG] Found document for ID {idx}")  # Debug 6
            else:
                print(f"[WARNING] No document for ID {idx}")  # Debug 7
        except Exception as e:
            print(f"[ERROR] Database query failed: {str(e)}")
    
    conn.close()
    print(f"[DEBUG] Context length: {len(context)}")  # Debug 8
    
    # Generate response
    try:
        context_str = '- ' + '\n- '.join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        print(f"[DEBUG] Final prompt:\n{prompt}")  # Debug 9
        
        response = perplexity_api_query(prompt)
        print("[DEBUG] Received API response")  # Debug 10
        return response
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        return ""

if __name__ == "__main__":
    print("=== Starting RAG Pipeline ===")
    result = rag_query("What is the ideal nitrogen content for wheat?")
    print("\n=== Final Result ===")
    print(result)