"""
Flock of Thought - Orkestrasi penalaran kolektif antara subsistem simbolik dan neural.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

# Import komponen yang sudah ada
from core.cognitive_core import CognitiveEngine
from core.neural_ecosystem import CognitiveKernel, AutonomousANN
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.chain_of_thought import ChainOfThought
from core.reflection import Reflection

# Import penyimpanan
from memory.storage import MemoryStore


@dataclass
class Thought:
    """Representasi satu pemikiran dalam rantai."""
    content: str
    vector: List[float]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # "symbolic", "neural", "chain"


class FlockOfThought:
    """
    Menggabungkan berbagai metode penalaran menjadi satu sistem koheren.
    Bertindak sebagai "konduktor" bagi CognitiveEngine, ANN, dan ChainOfThought.
    """
    
    def __init__(self, 
                 symbolic_dim: int = 64,
                 neural_topology: List[int] = [128, 64, 32],
                 use_persistence: bool = True):
        
        # Inisialisasi subsistem
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim, attention_heads=4)
        self.neural_kernel = CognitiveKernel(
            input_dim=neural_topology[0],
            hidden_dim=neural_topology[1],
            output_dim=neural_topology[2],
            symbolic_dim=symbolic_dim
        )
        self.chain_engine = ChainOfThought(self.symbolic_engine)
        self.neuromodulators = NeuromodulatorSystem()
        self.reflection = Reflection(self.neuromodulators)
        
        # Memori jangka panjang
        self.memory_store = MemoryStore() if use_persistence else None
        
        # Riwayat pemikiran
        self.thought_history: List[Thought] = []
        
        # Statistik
        self.cycle_count = 0
        self.cumulative_reward = 0.0
    
    def _text_to_vector(self, text: str, target_dim: int) -> List[float]:
        """Konversi teks ke vektor numerik (embedding sederhana)."""
        # Normalisasi dan padding
        vec = [0.0] * target_dim
        for i, ch in enumerate(text[:target_dim]):
            vec[i] = (ord(ch) % 100) / 100.0
        # Jika teks pendek, gunakan hash sisa
        if len(text) < target_dim:
            import hashlib
            hash_bytes = hashlib.md5(text.encode()).digest()
            for i in range(min(target_dim - len(text), len(hash_bytes))):
                vec[len(text) + i] = hash_bytes[i] / 255.0
        return vec
    
    def _vector_to_text(self, vec: List[float]) -> str:
        """Konversi vektor ke representasi string (untuk debugging)."""
        # Tidak ada inversi sempurna, cukup representasi numerik
        return f"vec[{vec[0]:.2f}, {vec[1]:.2f}, ...]"
    
    def process_stimulus(self, stimulus: str) -> Dict[str, Any]:
        """
        Memproses stimulus teks melalui semua jalur penalaran.
        Mengembalikan hasil gabungan beserta metadata.
        """
        self.cycle_count += 1
        
        # Konversi stimulus ke vektor untuk tiap subsistem
        stim_symbolic = self._text_to_vector(stimulus, self.symbolic_engine.dim)
        stim_neural = self._text_to_vector(stimulus, 128)  # Sesuaikan dengan input ANN
        
        # 1. Penalaran simbolik langsung
        symbolic_response = self.symbolic_engine.query(stim_symbolic)
        
        # 2. Penalaran melalui rantai pemikiran (Chain-of-Thought)
        cot_summary = self.chain_engine.reason(stimulus, steps=3)
        
        # 3. Penalaran neural (intuisi)
        neural_response = self.neural_kernel.process(stim_neural)
        
        # 4. Gabungkan respons dengan bobot yang dimodulasi oleh neuromodulator
        # Dopamine mempengaruhi bobot eksplorasi, Serotonin konservatisme
        dopamine = self.neuromodulators.dopamine.level
        serotonin = self.neuromodulators.serotonin.level
        
        w_symbolic = 0.4 * (1.0 + serotonin * 0.5)   # Serotonin tinggi -> lebih percaya simbolik
        w_neural = 0.4 * (1.0 + dopamine * 0.5)      # Dopamine tinggi -> lebih percaya neural
        w_chain = 0.2
        total_w = w_symbolic + w_neural + w_chain
        w_symbolic /= total_w
        w_neural /= total_w
        w_chain /= total_w
        
        # Kombinasi vektor (pastikan panjang sama)
        min_len = min(len(symbolic_response), len(neural_response), 32)
        combined_vec = [0.0] * min_len
        for i in range(min_len):
            combined_vec[i] = (
                w_symbolic * symbolic_response[i] +
                w_neural * neural_response[i] +
                w_chain * (cot_summary[i] if i < len(cot_summary) else 0.0)
            )
        
        # Hitung confidence berdasarkan agreement antar subsistem
        agreement = self._compute_agreement(symbolic_response, neural_response, cot_summary)
        
        # Catat pemikiran
        thought = Thought(
            content=self._vector_to_text(combined_vec),
            vector=combined_vec,
            confidence=agreement,
            source="ensemble"
        )
        self.thought_history.append(thought)
        
        # Evaluasi reflektif & update neuromodulator
        reflection_score = self.reflection.evaluate(
            [t.vector for t in self.thought_history[-5:]]
        )
        
        # Reward intrinsik berdasarkan confidence dan novelty
        reward = agreement * 0.5 + reflection_score * 0.5
        self.cumulative_reward += reward
        
        # Trigger neuromodulator events
        if agreement < 0.3:
            # Kebingungan tinggi -> tingkatkan noradrenaline (arousal)
            self.neuromodulators.update_all(
                deltas=NeuromodulatoryEvent.novelty_detection(0.7)
            )
        elif agreement > 0.8:
            # Keyakinan tinggi -> dopamine reward
            self.neuromodulators.update_all(
                deltas=NeuromodulatoryEvent.reward_prediction_error(reward, 0.5)
            )
        
        # Simpan ke memori jangka panjang jika perlu
        if self.memory_store and self.cycle_count % 10 == 0:
            self.memory_store.save_thought(thought)
        
        return {
            "response_vector": combined_vec,
            "confidence": agreement,
            "neuromodulator_levels": self.neuromodulators.get_all_levels(),
            "reflection_score": reflection_score,
            "symbolic_contribution": w_symbolic,
            "neural_contribution": w_neural,
            "chain_contribution": w_chain,
            "cycle": self.cycle_count
        }
    
    def _compute_agreement(self, vec1, vec2, vec3) -> float:
        """Menghitung kesepakatan antar vektor (cosine similarity rata-rata)."""
        def cosine_sim(a, b):
            dot = sum(ai*bi for ai, bi in zip(a, b))
            norm_a = sum(ai*ai for ai in a) ** 0.5
            norm_b = sum(bi*bi for bi in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        
        # Pastikan panjang sama
        min_len = min(len(vec1), len(vec2), len(vec3))
        v1 = vec1[:min_len]
        v2 = vec2[:min_len]
        v3 = vec3[:min_len]
        
        sim12 = cosine_sim(v1, v2)
        sim23 = cosine_sim(v2, v3)
        sim31 = cosine_sim(v3, v1)
        return (sim12 + sim23 + sim31) / 3.0
    
    def deliberate(self, problem: str, max_cycles: int = 5) -> str:
        """
        Melakukan penalaran mendalam terhadap suatu masalah.
        Mengembalikan kesimpulan dalam bentuk teks.
        """
        # Lakukan beberapa siklus pemrosesan
        for _ in range(max_cycles):
            result = self.process_stimulus(problem)
            
            # Jika confidence cukup tinggi, hentikan
            if result["confidence"] > 0.7:
                break
        
        # Hasil akhir dari state terakhir
        final_state = self.symbolic_engine.state
        # Konversi state ke teks (placeholder)
        response = self._state_to_natural_language(final_state)
        return response
    
    def _state_to_natural_language(self, state: List[float]) -> str:
        """[Placeholder] Konversi state vektor ke bahasa alami."""
        # Implementasi sederhana: ambil beberapa nilai tertinggi sebagai "konsep"
        top_indices = sorted(range(len(state)), key=lambda i: state[i], reverse=True)[:5]
        return f"Pemikiran terfokus pada dimensi {top_indices} dengan intensitas {[state[i] for i in top_indices]}"
    
    def save_state(self, filepath: str):
        """Menyimpan seluruh state sistem ke disk."""
        import pickle
        state_data = {
            "symbolic_memory": self.symbolic_engine.memory,
            "symbolic_state": self.symbolic_engine.state,
            "neural_weights": self.neural_kernel.ann.weights,
            "neuromodulator_levels": self.neuromodulators.get_all_levels(),
            "thought_history": [(t.content, t.vector, t.confidence) for t in self.thought_history[-100:]]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
    
    def load_state(self, filepath: str):
        """Memuat state sistem dari disk."""
        import pickle
        with open(filepath, 'rb') as f:
            state_data = pickle.load(f)
        self.symbolic_engine.memory = state_data["symbolic_memory"]
        self.symbolic_engine.state = state_data["symbolic_state"]
        self.neural_kernel.ann.weights = state_data["neural_weights"]
        # Tidak set neuromodulator langsung, biarkan beradaptasi