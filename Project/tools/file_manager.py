"""
File Manager Module (Backend Only)
---------------------------------
Modul Python murni untuk mengelola file dan direktori di Termux.
Menyediakan fungsi-fungsi dasar seperti menjelajah, membaca, menulis,
menghapus, menyalin, memindahkan, dan mencari file.
Dilengkapi dengan pembatasan akses direktori untuk keamanan.

Author: Assistant
Version: 1.0
"""

import os
import shutil
import fnmatch
import stat
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple


class FileManagerError(Exception):
    """Exception khusus untuk kesalahan operasi file manager."""
    pass


class FileManager:
    """
    Kelas utama pengelola file.
    
    Args:
        base_path (str, optional): Direktori root yang diizinkan untuk diakses.
                                   Jika None, akan menggunakan direktori home Termux
                                   (biasanya '/data/data/com.termux/files/home').
                                   Semua operasi akan dibatasi dalam direktori ini.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            # Default: home Termux
            self.base_path = os.path.realpath(os.path.expanduser("~"))
        else:
            self.base_path = os.path.realpath(os.path.expanduser(base_path))
        
        if not os.path.exists(self.base_path):
            raise FileManagerError(f"Base path tidak ditemukan: {self.base_path}")
        if not os.path.isdir(self.base_path):
            raise FileManagerError(f"Base path bukan direktori: {self.base_path}")
    
    def _resolve_path(self, path: str) -> str:
        """
        Mengubah path relatif menjadi absolut dan memastikan berada dalam base_path.
        
        Args:
            path (str): Path relatif atau absolut.
            
        Returns:
            str: Path absolut yang telah di-resolve.
            
        Raises:
            FileManagerError: Jika path berada di luar base_path atau tidak valid.
        """
        # Gabungkan dengan base_path jika relatif
        if not os.path.isabs(path):
            full_path = os.path.join(self.base_path, path)
        else:
            full_path = path
        
        # Resolve symlink dan normalisasi
        real_path = os.path.realpath(full_path)
        
        # Periksa apakah berada dalam base_path
        try:
            common = os.path.commonpath([real_path, self.base_path])
            if common != self.base_path:
                raise FileManagerError(
                    f"Akses ditolak: '{path}' berada di luar direktori yang diizinkan "
                    f"({self.base_path})"
                )
        except ValueError:
            # Terjadi jika path di drive berbeda (Windows), tidak relevan di Termux
            raise FileManagerError(f"Path tidak valid atau di luar batas: {path}")
        
        return real_path
    
    def _get_item_info(self, path: str) -> Dict[str, Union[str, int, bool]]:
        """
        Mendapatkan informasi detail suatu item (file/direktori).
        
        Args:
            path (str): Path absolut item.
            
        Returns:
            dict: Informasi item (nama, path, tipe, ukuran, permission, waktu modifikasi).
        """
        stat_info = os.stat(path)
        is_dir = os.path.isdir(path)
        info = {
            'name': os.path.basename(path),
            'path': path,
            'type': 'directory' if is_dir else 'file',
            'size': stat_info.st_size if not is_dir else 0,  # Ukuran direktori tidak langsung
            'permissions': stat.filemode(stat_info.st_mode),
            'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat() if hasattr(stat_info, 'st_ctime') else None,
            'is_symlink': os.path.islink(path)
        }
        if info['is_symlink']:
            info['symlink_target'] = os.readlink(path)
        return info
    
    # ==================== OPERASI DASAR ====================
    
    def list_dir(self, path: str = '.', show_hidden: bool = False) -> List[Dict]:
        """
        Mendaftar isi direktori.
        
        Args:
            path (str): Path direktori (relatif terhadap base_path).
            show_hidden (bool): Tampilkan file/direktori tersembunyi (diawali titik).
            
        Returns:
            list: Daftar item dengan informasi detail.
            
        Raises:
            FileManagerError: Jika path bukan direktori atau tidak dapat diakses.
        """
        abs_path = self._resolve_path(path)
        if not os.path.isdir(abs_path):
            raise FileManagerError(f"'{path}' bukan direktori")
        
        items = []
        try:
            for entry in os.scandir(abs_path):
                if not show_hidden and entry.name.startswith('.'):
                    continue
                items.append(self._get_item_info(entry.path))
        except PermissionError:
            raise FileManagerError(f"Izin ditolak saat membaca direktori: {path}")
        
        # Urutkan: direktori dulu, lalu file, berdasarkan nama
        items.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))
        return items
    
    def get_info(self, path: str) -> Dict:
        """
        Mendapatkan informasi detail suatu file/direktori.
        
        Args:
            path (str): Path item.
            
        Returns:
            dict: Informasi item.
        """
        abs_path = self._resolve_path(path)
        if not os.path.exists(abs_path):
            raise FileManagerError(f"'{path}' tidak ditemukan")
        return self._get_item_info(abs_path)
    
    def exists(self, path: str) -> bool:
        """Memeriksa apakah path ada."""
        try:
            self._resolve_path(path)
            return os.path.exists(self._resolve_path(path))
        except FileManagerError:
            return False
    
    def is_file(self, path: str) -> bool:
        """Memeriksa apakah path adalah file."""
        try:
            abs_path = self._resolve_path(path)
            return os.path.isfile(abs_path)
        except FileManagerError:
            return False
    
    def is_dir(self, path: str) -> bool:
        """Memeriksa apakah path adalah direktori."""
        try:
            abs_path = self._resolve_path(path)
            return os.path.isdir(abs_path)
        except FileManagerError:
            return False
    
    def get_size(self, path: str) -> int:
        """
        Mendapatkan ukuran file/direktori dalam byte.
        Untuk direktori, menghitung total ukuran isinya secara rekursif.
        """
        abs_path = self._resolve_path(path)
        if os.path.isfile(abs_path):
            return os.path.getsize(abs_path)
        elif os.path.isdir(abs_path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(abs_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp) and not os.path.islink(fp):
                        total += os.path.getsize(fp)
            return total
        else:
            raise FileManagerError(f"'{path}' bukan file atau direktori")
    
    # ==================== MEMBACA & MENULIS ====================
    
    def read_file(self, path: str, binary: bool = False) -> Union[str, bytes]:
        """
        Membaca isi file.
        
        Args:
            path (str): Path file.
            binary (bool): Jika True, baca sebagai binary (mengembalikan bytes).
            
        Returns:
            str atau bytes: Konten file.
        """
        abs_path = self._resolve_path(path)
        if not os.path.isfile(abs_path):
            raise FileManagerError(f"'{path}' bukan file atau tidak ditemukan")
        
        mode = 'rb' if binary else 'r'
        try:
            with open(abs_path, mode) as f:
                return f.read()
        except UnicodeDecodeError:
            raise FileManagerError(f"File '{path}' bukan teks, gunakan binary=True")
        except Exception as e:
            raise FileManagerError(f"Gagal membaca file: {e}")
    
    def write_file(self, path: str, content: Union[str, bytes], append: bool = False) -> None:
        """
        Menulis konten ke file.
        
        Args:
            path (str): Path file.
            content (str|bytes): Konten yang akan ditulis.
            append (bool): Jika True, tambahkan ke akhir file.
        """
        abs_path = self._resolve_path(path)
        mode = 'ab' if append else 'wb' if isinstance(content, bytes) else 'a' if append else 'w'
        
        # Pastikan direktori induk ada
        parent = os.path.dirname(abs_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        
        try:
            if isinstance(content, bytes):
                with open(abs_path, mode) as f:
                    f.write(content)
            else:
                with open(abs_path, mode, encoding='utf-8') as f:
                    f.write(content)
        except Exception as e:
            raise FileManagerError(f"Gagal menulis file: {e}")
    
    def append_file(self, path: str, content: Union[str, bytes]) -> None:
        """Menambahkan konten ke akhir file."""
        self.write_file(path, content, append=True)
    
    # ==================== MANIPULASI ====================
    
    def delete(self, path: str, recursive: bool = False) -> None:
        """
        Menghapus file atau direktori.
        
        Args:
            path (str): Path yang akan dihapus.
            recursive (bool): Jika True, hapus direktori beserta isinya.
                             Jika False dan path adalah direktori tidak kosong, akan error.
        """
        abs_path = self._resolve_path(path)
        if not os.path.exists(abs_path):
            raise FileManagerError(f"'{path}' tidak ditemukan")
        
        try:
            if os.path.isfile(abs_path) or os.path.islink(abs_path):
                os.remove(abs_path)
            elif os.path.isdir(abs_path):
                if recursive:
                    shutil.rmtree(abs_path)
                else:
                    os.rmdir(abs_path)  # Hanya direktori kosong
        except OSError as e:
            raise FileManagerError(f"Gagal menghapus '{path}': {e}")
    
    def rename(self, old_path: str, new_name: str) -> str:
        """
        Mengganti nama file/direktori (dalam direktori yang sama).
        
        Args:
            old_path (str): Path item yang akan diganti nama.
            new_name (str): Nama baru (bukan path lengkap).
            
        Returns:
            str: Path baru.
        """
        abs_old = self._resolve_path(old_path)
        if not os.path.exists(abs_old):
            raise FileManagerError(f"'{old_path}' tidak ditemukan")
        
        parent = os.path.dirname(abs_old)
        new_path = os.path.join(parent, new_name)
        abs_new = self._resolve_path(new_path)
        
        try:
            os.rename(abs_old, abs_new)
        except OSError as e:
            raise FileManagerError(f"Gagal mengganti nama: {e}")
        return abs_new
    
    def move(self, src: str, dest: str) -> None:
        """
        Memindahkan file/direktori ke lokasi baru.
        
        Args:
            src (str): Path sumber.
            dest (str): Path tujuan (file atau direktori).
        """
        abs_src = self._resolve_path(src)
        if not os.path.exists(abs_src):
            raise FileManagerError(f"Sumber '{src}' tidak ditemukan")
        
        abs_dest = self._resolve_path(dest)
        try:
            shutil.move(abs_src, abs_dest)
        except Exception as e:
            raise FileManagerError(f"Gagal memindahkan: {e}")
    
    def copy(self, src: str, dest: str) -> None:
        """
        Menyalin file/direktori.
        
        Args:
            src (str): Path sumber.
            dest (str): Path tujuan.
        """
        abs_src = self._resolve_path(src)
        if not os.path.exists(abs_src):
            raise FileManagerError(f"Sumber '{src}' tidak ditemukan")
        
        abs_dest = self._resolve_path(dest)
        try:
            if os.path.isdir(abs_src):
                shutil.copytree(abs_src, abs_dest)
            else:
                # Pastikan direktori tujuan ada
                os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
                shutil.copy2(abs_src, abs_dest)
        except Exception as e:
            raise FileManagerError(f"Gagal menyalin: {e}")
    
    def create_dir(self, path: str, exist_ok: bool = True) -> None:
        """
        Membuat direktori baru.
        
        Args:
            path (str): Path direktori yang akan dibuat.
            exist_ok (bool): Jika True, tidak error jika direktori sudah ada.
        """
        abs_path = self._resolve_path(path)
        try:
            os.makedirs(abs_path, exist_ok=exist_ok)
        except FileExistsError:
            raise FileManagerError(f"Direktori '{path}' sudah ada")
        except Exception as e:
            raise FileManagerError(f"Gagal membuat direktori: {e}")
    
    # ==================== PENCARIAN ====================
    
    def search(self, pattern: str, path: str = '.', recursive: bool = True,
               case_sensitive: bool = False) -> List[str]:
        """
        Mencari file/direktori berdasarkan pola nama (wildcard).
        
        Args:
            pattern (str): Pola pencarian (contoh: '*.py', 'test*').
            path (str): Direktori awal pencarian.
            recursive (bool): Jika True, cari di subdirektori.
            case_sensitive (bool): Jika True, pencarian sensitif huruf besar/kecil.
            
        Returns:
            list: Daftar path item yang cocok.
        """
        abs_path = self._resolve_path(path)
        if not os.path.isdir(abs_path):
            raise FileManagerError(f"'{path}' bukan direktori")
        
        results = []
        flags = 0 if case_sensitive else fnmatch.IGNORECASE if hasattr(fnmatch, 'IGNORECASE') else 0
        
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                # Cek direktori
                for d in dirs:
                    if fnmatch.fnmatch(d, pattern) if case_sensitive else fnmatch.fnmatch(d.lower(), pattern.lower()):
                        results.append(os.path.join(root, d))
                # Cek file
                for f in files:
                    if fnmatch.fnmatch(f, pattern) if case_sensitive else fnmatch.fnmatch(f.lower(), pattern.lower()):
                        results.append(os.path.join(root, f))
        else:
            with os.scandir(abs_path) as entries:
                for entry in entries:
                    name = entry.name
                    match = fnmatch.fnmatch(name, pattern) if case_sensitive else fnmatch.fnmatch(name.lower(), pattern.lower())
                    if match:
                        results.append(entry.path)
        
        return results
    
    # ==================== UTILITAS ====================
    
    def get_tree(self, path: str = '.', max_depth: int = 3, show_hidden: bool = False) -> str:
        """
        Menghasilkan representasi pohon direktori dalam bentuk teks.
        
        Args:
            path (str): Direktori awal.
            max_depth (int): Kedalaman maksimum.
            show_hidden (bool): Tampilkan item tersembunyi.
            
        Returns:
            str: String pohon direktori.
        """
        abs_path = self._resolve_path(path)
        if not os.path.isdir(abs_path):
            raise FileManagerError(f"'{path}' bukan direktori")
        
        lines = []
        prefix = ""
        
        def _tree(current_path, prefix, depth):
            if depth > max_depth:
                return
            entries = sorted(os.scandir(current_path), key=lambda e: e.name)
            filtered = [e for e in entries if show_hidden or not e.name.startswith('.')]
            for i, entry in enumerate(filtered):
                is_last = i == len(filtered) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir(follow_symlinks=False):
                    extension = "    " if is_last else "│   "
                    _tree(entry.path, prefix + extension, depth + 1)
        
        lines.append(os.path.basename(abs_path) or abs_path)
        _tree(abs_path, "", 1)
        return "\n".join(lines)
    
    def get_disk_usage(self) -> Dict[str, int]:
        """
        Mendapatkan informasi penggunaan disk untuk filesystem tempat base_path berada.
        
        Returns:
            dict: total, used, free dalam byte.
        """
        usage = shutil.disk_usage(self.base_path)
        return {
            'total': usage.total,
            'used': usage.used,
            'free': usage.free
        }


# ==================== CONTOH PENGGUNAAN ====================
if __name__ == "__main__":
    # Contoh penggunaan modul (hanya untuk pengujian)
    fm = FileManager()
    
    print("=== File Manager Test ===")
    print(f"Base path: {fm.base_path}")
    
    # List direktori home
    print("\nIsi direktori home:")
    for item in fm.list_dir(show_hidden=False)[:5]:  # Batasi 5 item
        print(f"  {item['name']} ({item['type']}) - {item['size']} bytes")
    
    # Info penggunaan disk
    disk = fm.get_disk_usage()
    print(f"\nDisk usage: {disk['used']//(1024**2)} MB / {disk['total']//(1024**2)} MB")