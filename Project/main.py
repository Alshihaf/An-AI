#!/usr/bin/env python3
"""
Samre - Autonomous AI with RL/AGI aspirations
Entry point untuk menjalankan sistem.
"""

import sys
import os
import time
import argparse
from typing import Optional

# Tambahkan path proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flock_of_thought import FlockOfThought
from tools.file_manager import FileManager


class SamreCLI:
    """Antarmuka command-line untuk berinteraksi dengan Samre."""
    
    def __init__(self, use_persistence: bool = True):
        print("🧠 Menginisialisasi Samre...")
        self.flock = FlockOfThought(use_persistence=use_persistence)
        self.file_manager = FileManager()  # Untuk akses file system jika diperlukan
        self.running = True
        print("✅ Samre siap. Ketik 'help' untuk bantuan, 'exit' untuk keluar.\n")
    
    def print_status(self):
        levels = self.flock.neuromodulators.get_all_levels()
        print("\n📊 Status Neuromodulator:")
        for name, level in levels.items():
            bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
            print(f"  {name:15} [{bar}] {level:.2f}")
        print(f"  Siklus: {self.flock.cycle_count} | Reward Kumulatif: {self.flock.cumulative_reward:.3f}\n")
    
    def process_command(self, user_input: str):
        if user_input.lower() in ["exit", "quit"]:
            self.running = False
            print("👋 Menyimpan state dan keluar...")
            self.flock.save_state("samre_state.pkl")
            if self.flock.memory_store:
                self.flock.memory_store.close()
            return
        
        if user_input.lower() == "help":
            print("""
Perintah yang tersedia:
  [teks apa pun]  - Berbicara dengan Samre, memicu penalaran.
  status          - Tampilkan level neuromodulator dan statistik.
  think [teks]    - Lakukan Chain-of-Thought mendalam pada topik.
  save [file]     - Simpan state saat ini ke file.
  load [file]     - Muat state dari file.
  memory [query]  - Cari memori serupa (jika ada query).
  exit/quit       - Keluar.
            """)
            return
        
        if user_input.lower().startswith("status"):
            self.print_status()
            return
        
        if user_input.lower().startswith("save"):
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "samre_state.pkl"
            self.flock.save_state(filename)
            print(f"✅ State disimpan ke {filename}")
            return
        
        if user_input.lower().startswith("load"):
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "samre_state.pkl"
            try:
                self.flock.load_state(filename)
                print(f"✅ State dimuat dari {filename}")
            except FileNotFoundError:
                print(f"❌ File {filename} tidak ditemukan.")
            return
        
        if user_input.lower().startswith("think"):
            # Mode penalaran mendalam
            topic = user_input[5:].strip()
            if not topic:
                topic = "keberadaan dan tujuan"
            print(f"🤔 Merenungkan: '{topic}'...")
            conclusion = self.flock.deliberate(topic, max_cycles=10)
            print(f"💡 Kesimpulan: {conclusion}")
            return
        
        if user_input.lower().startswith("memory"):
            parts = user_input.split(maxsplit=1)
            query = parts[1] if len(parts) > 1 else ""
            if self.flock.memory_store:
                if query:
                    # Konversi query ke vektor
                    q_vec = self.flock._text_to_vector(query, self.flock.symbolic_engine.dim)
                    similar = self.flock.memory_store.retrieve_similar_thoughts(q_vec, top_k=3)
                    print("🔍 Pemikiran serupa:")
                    for item in similar:
                        print(f"  [{item['similarity']:.2f}] {item['content'][:60]}...")
                else:
                    print("ℹ️ Gunakan 'memory [kata kunci]' untuk mencari.")
            else:
                print("❌ Fitur memori tidak diaktifkan.")
            return
        
        # Default: proses sebagai stimulus
        print("🧠 Memproses...")
        start = time.time()
        result = self.flock.process_stimulus(user_input)
        elapsed = time.time() - start
        
        print(f"\n✨ Respons (confidence: {result['confidence']:.2f})")
        print(f"   Vektor gabungan: {result['response_vector'][:5]}...")
        print(f"   Kontribusi: Simbolik {result['symbolic_contribution']:.2f} | Neural {result['neural_contribution']:.2f} | CoT {result['chain_contribution']:.2f}")
        print(f"   Waktu: {elapsed:.3f}s")
    
    def run(self):
        """Loop utama CLI."""
        try:
            while self.running:
                try:
                    user_input = input("\n💬 > ").strip()
                    if not user_input:
                        continue
                    self.process_command(user_input)
                except KeyboardInterrupt:
                    print("\n⚠️ Gunakan 'exit' untuk keluar.")
                except Exception as e:
                    print(f"❌ Error: {e}")
        finally:
            if hasattr(self, 'flock') and self.flock.memory_store:
                self.flock.memory_store.close()


def main():
    parser = argparse.ArgumentParser(description="Samre - Autonomous AI")
    parser.add_argument("--no-persistence", action="store_true", help="Nonaktifkan penyimpanan memori jangka panjang")
    parser.add_argument("--load", type=str, help="Muat state dari file saat startup")
    args = parser.parse_args()
    
    cli = SamreCLI(use_persistence=not args.no_persistence)
    if args.load:
        try:
            cli.flock.load_state(args.load)
            print(f"✅ State dimuat dari {args.load}")
        except Exception as e:
            print(f"❌ Gagal memuat state: {e}")
    
    cli.run()


if __name__ == "__main__":
    main()