"""
Memory Storage - Penyimpanan persisten untuk pengalaman dan pengetahuan.
"""

import os
import json
import sqlite3
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np


class MemoryStore:
    """
    Mengelola penyimpanan memori jangka panjang Samre.
    Menggunakan SQLite untuk metadata dan file untuk vektor besar.
    """
    
    def __init__(self, db_path: str = "samre_memory.db", vector_dir: str = "memory_vectors"):
        self.db_path = db_path
        self.vector_dir = vector_dir
        os.makedirs(vector_dir, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                content TEXT,
                confidence REAL,
                source TEXT,
                vector_file TEXT,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                summary TEXT,
                reward_total REAL,
                tags TEXT
            )
        """)
        self.conn.commit()
    
    def save_thought(self, thought) -> int:
        """Menyimpan satu objek Thought ke database dan file vektor."""
        import json
        
        # Simpan vektor ke file terpisah
        vector_filename = f"thought_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
        vector_path = os.path.join(self.vector_dir, vector_filename)
        with open(vector_path, 'wb') as f:
            pickle.dump(thought.vector, f)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO thoughts (timestamp, content, confidence, source, vector_file, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            thought.timestamp,
            thought.content,
            thought.confidence,
            thought.source,
            vector_filename,
            json.dumps({})  # Bisa ditambah metadata tambahan
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def retrieve_similar_thoughts(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Mencari pemikiran yang mirip berdasarkan cosine similarity.
        (Implementasi sederhana, untuk produksi gunakan FAISS atau Annoy)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, vector_file, content, confidence FROM thoughts ORDER BY timestamp DESC LIMIT 1000")
        rows = cursor.fetchall()
        
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        
        for row in rows:
            thought_id, vec_file, content, conf = row
            vec_path = os.path.join(self.vector_dir, vec_file)
            try:
                with open(vec_path, 'rb') as f:
                    vec = pickle.load(f)
                # Hitung cosine similarity
                dot = np.dot(query_vector, vec)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    sim = dot / (query_norm * norm)
                else:
                    sim = 0.0
                similarities.append((sim, thought_id, content, conf))
            except FileNotFoundError:
                continue
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top = similarities[:top_k]
        return [{"similarity": s, "id": tid, "content": c, "confidence": cf} for s, tid, c, cf in top]
    
    def save_episode(self, start_time: float, end_time: float, summary: str, reward: float, tags: List[str]):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episodes (start_time, end_time, summary, reward_total, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (start_time, end_time, summary, reward, ",".join(tags)))
        self.conn.commit()
    
    def close(self):
        self.conn.close()