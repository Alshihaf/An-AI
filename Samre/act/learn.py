"""
Learning Actuator - Mekanisme untuk pembelajaran dan sintesis pengetahuan.
----------------------------------------------------------------------
Modul ini memungkinkan Samre untuk secara aktif belajar dari file
dan mensintesis pengetahuan baru dari proses pemikirannya sendiri.

Author: Assistant
Version: 1.0
"""

import math
from typing import TYPE_CHECKING, List

# Hindari circular import dengan forward reference
if TYPE_CHECKING:
    from core.flock_of_thought import FlockOfThought, Thought
    from tools.file_manager import FileManager

class LearningActuatorError(Exception):
    """Exception khusus untuk kesalahan pada proses belajar."""
    pass

class LearningActuator:
    """
    Mengorkestrasi tindakan belajar, baik dari sumber eksternal (file)
    maupun dari sintesis internal (pemikiran).
    """
    def __init__(self, flock: 'FlockOfThought', file_manager: 'FileManager'):
        """
        Inisialisasi LearningActuator.

        Args:
            flock (FlockOfThought): Instance dari otak kognitif utama.
            file_manager (FileManager): Instance untuk interaksi dengan sistem file.
        """
        self.flock = flock
        self.file_manager = file_manager

    def learn_from_file(self, path: str) -> str:
        """
        Membaca file, memecahnya menjadi potongan pengetahuan, dan menyimpannya ke memori.

        Args:
            path (str): Path ke file yang akan dipelajari.

        Returns:
            str: Ringkasan hasil proses pembelajaran.
        """
        try:
            print(f"🧠 Memulai proses belajar dari file: {path}...")
            content = self.file_manager.read_file(path)
            
            if isinstance(content, bytes):
                raise LearningActuatorError("Tidak dapat belajar dari file biner.")
            
            if not content.strip():
                return f"File '{path}' kosong, tidak ada yang dipelajari."

            # Pecah konten menjadi paragraf atau potongan yang lebih kecil
            # Ini lebih baik daripada mempelajari satu file besar sekaligus
            chunks = [chunk for chunk in content.split('\n\n') if chunk.strip()]
            
            if not chunks:
                chunks = [content] # Jika tidak ada paragraf, pelajari seluruh file

            num_chunks = len(chunks)
            learned_count = 0

            for i, chunk in enumerate(chunks):
                # Ubah potongan teks menjadi "pemikiran" yang dapat disimpan
                # Kita gunakan source "learning" untuk menandai asal-usul pengetahuan ini
                thought_vector = self.flock._text_to_vector(chunk, self.flock.symbolic_engine.dim)
                
                # Hanya simpan jika vektornya tidak kosong
                if any(v != 0 for v in thought_vector):
                    new_thought: 'Thought' = self.flock.thought_history.append(
                        self.flock.Thought(
                            content=chunk,
                            vector=thought_vector,
                            confidence=0.85,  # Keyakinan cukup tinggi karena dari sumber yang dipelajari
                            source="learning"
                        )
                    )
                    
                    # Simpan ke memori jangka panjang jika tersedia
                    if self.flock.memory_store:
                        self.flock.memory_store.save_thought(new_thought)
                    
                    learned_count += 1
                print(f"  -> Memproses potongan {i+1}/{num_chunks}...")

            return f"✅ Berhasil mempelajari {learned_count} dari {num_chunks} potongan pengetahuan dari file '{path}'."

        except Exception as e:
            raise LearningActuatorError(f"Gagal belajar dari file '{path}': {e}")

    def synthesize_and_learn(self, topic: str, new_file_path: str) -> str:
        """
        Berpikir mendalam tentang suatu topik, menulis hasilnya ke file baru,
        lalu mempelajari file yang baru dibuat itu.

        Args:
            topic (str): Topik untuk direnungkan.
            new_file_path (str): Path file untuk menyimpan hasil sintesis.

        Returns:
            str: Ringkasan hasil proses sintesis dan pembelajaran.
        """
        try:
            print(f"🤔 Mensintesis pengetahuan tentang topik: '{topic}'...")
            
            # 1. Berpikir mendalam (deliberate) untuk menghasilkan pengetahuan baru
            synthesized_knowledge = self.flock.deliberate(topic, max_cycles=7)
            
            if not synthesized_knowledge or "terfokus pada dimensi" in synthesized_knowledge:
                 return f"Gagal menghasilkan sintesis yang bermakna untuk topik: '{topic}'."

            header = f"Sintesis Pengetahuan oleh Samre\nTopik: {topic}\n"
            content_to_write = header + "="*20 + "\n" + synthesized_knowledge
            
            # 2. Menulis pengetahuan baru ke file
            print(f"✍️ Menulis sintesis ke file baru: {new_file_path}...")
            self.file_manager.write_file(new_file_path, content_to_write)
            
            # 3. Belajar dari file yang baru saja dibuat (siklus tertutup)
            learning_result = self.learn_from_file(new_file_path)
            
            return f"🎉 Proses sintesis selesai. {learning_result}"

        except Exception as e:
            raise LearningActuatorError(f"Gagal saat sintesis dan belajar: {e}")
