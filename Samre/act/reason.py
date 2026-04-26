"""
Reasoning Actuator - Mekanisme untuk penalaran deduktif dan sintesis.
-----------------------------------------------------------------------
Modul ini memungkinkan Samre untuk secara aktif menalar berdasarkan
berbagai sumber informasi, menghasilkan pengetahuan baru, dan secara
opsional menyimpannya kembali ke dalam basis pengetahuannya.

Author: Assistant
Version: 1.0
"""

import time
from typing import TYPE_CHECKING, List

# Hindari circular import dengan forward reference
if TYPE_CHECKING:
    from core.flock_of_thought import FlockOfThought
    from tools.file_manager import FileManager
    from act.learn import LearningActuator

class ReasoningActuatorError(Exception):
    """Exception khusus untuk kesalahan pada proses penalaran."""
    pass

class ReasoningActuator:
    """
    Mengorkestrasi tindakan penalaran kompleks, menghubungkan pemahaman,
    pemikiran mendalam, dan sintesis pengetahuan.
    """
    def __init__(self, flock: 'FlockOfThought', file_manager: 'FileManager', learning_actuator: 'LearningActuator'):
        """
        Inisialisasi ReasoningActuator.

        Args:
            flock (FlockOfThought): Instance dari otak kognitif utama.
            file_manager (FileManager): Instance untuk interaksi dengan sistem file.
            learning_actuator (LearningActuator): Instance untuk aksi pembelajaran.
        """
        self.flock = flock
        self.file_manager = file_manager
        self.learning_actuator = learning_actuator

    def deduce_from_sources(self, topic: str, source_paths: List[str]) -> str:
        """
        Menyerap pengetahuan dari beberapa file sumber untuk menalar tentang suatu topik.
        Ini adalah proses penalaran "sementara" tanpa menyimpan hasilnya.

        Args:
            topic (str): Topik utama untuk dinalar.
            source_paths (List[str]): Daftar path file yang menjadi dasar penalaran.

        Returns:
            str: Hasil penalaran atau deduksi.
        """
        try:
            print(f"🤔 Memulai penalaran deduktif tentang '{topic}'...")
            
            # 1. Temporarily learn from sources to build context
            # Ini akan memasukkan pengetahuan dari file ke dalam "memori jangka pendek" Samre
            for path in source_paths:
                print(f"   -> Menyerap konteks dari: {path}")
                # Kita tidak menggunakan learn_from_file secara langsung untuk menghindari penyimpanan permanen
                content = self.file_manager.read_file(path)
                if not isinstance(content, bytes):
                    # Buat "pemikiran" sementara untuk mempengaruhi penalaran berikutnya
                    temp_thought_vec = self.flock._text_to_vector(content, self.flock.symbolic_engine.dim)
                    temp_thought = self.flock.Thought(content, temp_thought_vec, 0.8, source="context")
                    self.flock.thought_history.append(temp_thought)
            
            # 2. Deliberate on the topic with the new context
            # Penalaran mendalam sekarang akan sangat dipengaruhi oleh konteks yang baru diserap
            print(f"   -> Melakukan penalaran mendalam...")
            deduction = self.flock.deliberate(f"Berdasarkan konteks yang baru diberikan, buat kesimpulan tentang: {topic}", max_cycles=10)
            
            if not deduction or "terfokus pada dimensi" in deduction:
                return f"Gagal melakukan deduksi yang bermakna untuk topik: '{topic}' dari sumber yang diberikan."

            return f"Deduksi tentang '{topic}':\n{deduction}"
        
        except Exception as e:
            raise ReasoningActuatorError(f"Gagal saat melakukan deduksi: {e}")

    def reason_and_synthesize(self, topic: str, source_paths: List[str], new_file_path: str) -> str:
        """
        Melakukan penalaran dari sumber, menulis hasilnya ke file baru,
        dan kemudian mempelajari file tersebut. Ini adalah siklus belajar-menalar penuh.

        Args:
            topic (str): Topik utama.
            source_paths (List[str]): Daftar file sumber.
            new_file_path (str): File untuk menyimpan hasil sintesis.

        Returns:
            str: Ringkasan seluruh proses.
        """
        try:
            # 1. Lakukan deduksi untuk menghasilkan pengetahuan baru
            deduced_knowledge = self.deduce_from_sources(topic, source_paths)
            
            # Hapus bagian prefix untuk mendapatkan konten bersih
            clean_knowledge = deduced_knowledge.replace(f"Deduksi tentang '{topic}':\n", "")
            
            # 2. Tulis pengetahuan hasil sintesis ke file
            print(f"✍️ Menulis hasil penalaran ke file: {new_file_path}...")
            header = f"Sintesis Penalaran oleh Samre\nTopik: {topic}\nSumber: {', '.join(source_paths)}\n"
            content_to_write = header + "="*20 + "\n" + clean_knowledge
            self.file_manager.write_file(new_file_path, content_to_write)
            
            # 3. Gunakan LearningActuator untuk mempelajari pengetahuan baru secara permanen
            print(f"🧠 Mempelajari pengetahuan yang baru disintesis...")
            learning_result = self.learning_actuator.learn_from_file(new_file_path)
            
            return f"🎉 Proses penalaran dan sintesis selesai. {learning_result}"
        except Exception as e:
            raise ReasoningActuatorError(f"Gagal saat penalaran dan sintesis: {e}")

