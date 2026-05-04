"""
FileManager - Jembatan antara Agen dan Sistem File Fisik.

Kelas ini menyediakan antarmuka terstruktur untuk berinteraksi dengan 
sistem file, menggunakan alat (API) yang tersedia di lingkungan.
"""
from typing import Dict, List

# Ini adalah placeholder untuk mensimulasikan panggilan API yang sebenarnya.
# Dalam implementasi nyata, ini akan menjadi modul API Anda.
from default_api import read_file, write_file, list_files

class FileManager:
    def __init__(self):
        """Menginisialisasi FileManager."""
        print("📦 FileManager diinisialisasi.")

    def read(self, file_path: str) -> str:
        """
        Membaca konten dari sebuah file.

        Args:
            file_path: Path relatif ke file yang akan dibaca.

        Returns:
            Konten file sebagai string, atau string kosong jika gagal.
        """
        print(f"📂 Membaca file: {file_path}")
        response = read_file(path=file_path)
        if response["status"] == "succeeded":
            return response["result"]
        else:
            print(f"GAGAL membaca file {file_path}: {response.get('result')}")
            return ""

    def write(self, file_path: str, content: str) -> bool:
        """
        Menulis atau menimpa konten ke sebuah file.

        Args:
            file_path: Path relatif ke file yang akan ditulis.
            content: Konten yang akan ditulis ke file.

        Returns:
            True jika berhasil, False jika gagal.
        """
        print(f"✍️ Menulis ke file: {file_path}")
        response = write_file(path=file_path, content=content)
        if response["status"] == "succeeded":
            return True
        else:
            print(f"GAGAL menulis ke file {file_path}: {response.get('result')}")
            return False

    def list_all(self, path: str = ".") -> List[str]:
        """
        Mendaftar semua file dan direktori dalam path yang diberikan.

        Args:
            path: Path relatif ke direktori yang akan didaftar.

        Returns:
            Daftar nama file dan direktori, atau daftar kosong jika gagal.
        """
        print(f"📋 Mendaftar file di: {path}")
        response = list_files(path=path)
        if response["status"] == "succeeded":
            # Asumsi respons API mengembalikan daftar file dalam format tertentu
            # Di sini kita sesuaikan dengan format yang diharapkan (misalnya, list of strings)
            # Ini mungkin perlu disesuaikan berdasarkan output API yang sebenarnya.
            return response.get("result", [])
        else:
            print(f"GAGAL mendaftar file di {path}: {response.get('result')}")
            return []
