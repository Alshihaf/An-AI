"""
Actuators - Komponen Pelaksana Tindakan untuk Samre.

Setiap kelas di sini mewakili kemampuan agen untuk melakukan tindakan tertentu
di dalam lingkungannya, seperti belajar, bernalar, atau berevolusi.
"""

from core.file_manager import FileManager
from typing import Dict, Any

class LearningActuator:
    def __init__(self, file_manager: FileManager, knowledge_base: Dict[str, Any]):
        self.file_manager = file_manager
        self.knowledge_base = knowledge_base

    def execute(self, target_file: str = "README.md") -> bool:
        """
        Menjalankan tindakan BELAJAR.
        Membaca file dan mengintegrasikan informasinya ke dalam basis pengetahuan.
        """
        print(f"📚 LEARNING: Mencoba belajar dari '{target_file}'.")
        content = self.file_manager.read(target_file)
        if content:
            self.knowledge_base[target_file] = {
                "content": content,
                "lines": len(content.splitlines())
            }
            print(f"    ✅ Berhasil: Pengetahuan tentang '{target_file}' telah diperbarui.")
            return True
        else:
            print(f"    ❌ Gagal: Tidak dapat membaca '{target_file}' untuk belajar.")
            return False

class ReasoningActuator:
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge_base = knowledge_base

    def execute(self) -> bool:
        """
        Menjalankan tindakan BERNALAR.
        Menganalisis basis pengetahuan untuk menarik kesimpulan sederhana.
        """
        print("🤔 REASONING: Menganalisis basis pengetahuan...")
        if not self.knowledge_base:
            print("    ❌ Gagal: Basis pengetahuan kosong. Tidak ada yang bisa dinalar.")
            return False

        readme_knowledge = self.knowledge_base.get("README.md")
        if readme_knowledge and "otonom" in readme_knowledge.get("content", ""):
            print("    ✅ Kesimpulan: Berdasarkan README.md, proyek ini adalah agen otonom.")
            return True
        else:
            print("    ⚠️ Tidak dapat menarik kesimpulan yang kuat dari pengetahuan saat ini.")
            return False

class EvolutionaryActuator:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def execute(self, target_file: str = "Samre/core/sws_logic.py") -> bool:
        """
        Menjalankan tindakan EVOLVE (langkah pertama: analisis).
        Membaca kode sumbernya sendiri untuk mempersiapkan modifikasi.
        """
        print(f"🧬 EVOLVING: Mempersiapkan evolusi dengan menganalisis '{target_file}'.")
        source_code = self.file_manager.read(target_file)
        if source_code:
            print(f"    ✅ Analisis Awal: Berhasil membaca {len(source_code)} karakter kode sumber.")
            # Di masa depan, langkah ini akan melibatkan analisis yang lebih dalam (misalnya, AST)
            # dan menghasilkan proposal perubahan kode.
            print("    下一步 (Langkah selanjutnya): Usulkan dan terapkan perubahan kode.")
            return True
        else:
            print(f"    ❌ Gagal: Tidak dapat membaca kode sumber '{target_file}' untuk evolusi.")
            return False
