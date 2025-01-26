import faiss
import numpy as np

class FAISSIndex:
    def __init__(self):
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def add(self, embedding, idx):
        self.index.add(np.array([embedding]).astype('float32'))
        
    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        return indices[0]
    
    def save(self, path):
        faiss.write_index(self.index, path)
    
    @classmethod
    def load(cls, path):
        instance = cls()
        instance.index = faiss.read_index(path)
        return instance