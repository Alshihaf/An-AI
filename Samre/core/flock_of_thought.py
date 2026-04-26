"""
Flock of Thought - Otak kognitif utama Samre.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import re
import hashlib
import math
import json

# Import semua komponen arsitektur
from core.cognitive_core import CognitiveEngine
from core.neural_ecosystem import CognitiveKernel, AutonomousANN
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.chain_of_thought import ChainOfThought
from core.reflection import Reflection
from memory.storage import MemoryStore
from tools.file_manager import FileManager
from act.learn import LearningActuator
from act.reason import ReasoningActuator # <- Import baru

@dataclass
class Tool:
    name: str; description: str; function: Callable; parameters: List[str]

class ToolOrchestrator:
    """Mengelola, memilih, dan mengeksekusi semua alat dan tindakan yang tersedia."""
    def __init__(self, flock: 'FlockOfThought'):
        self.flock = flock
        self.tools: Dict[str, Tool] = {}
        self.tool_vectors: Dict[str, List[float]] = {}
        
        # Inisialisasi semua modul fungsional
        self.file_manager = FileManager(base_path='.')
        self.learning_actuator = LearningActuator(flock, self.file_manager)
        self.reasoning_actuator = ReasoningActuator(flock, self.file_manager, self.learning_actuator) # <- Inisialisasi baru
        
        self._register_all_tools()

    def _register_all_tools(self):
        """Mendaftarkan semua alat dari semua modul aktuator."""
        all_tools = {
            # Alat dari FileManager
            "list_files": Tool("list_files", "Melihat dan mendaftar file di sebuah direktori.", self.file_manager.list_dir, ["path"]),
            "read_file": Tool("read_file", "Membaca dan menampilkan isi dari sebuah file.", self.file_manager.read_file, ["path"]),
            "write_file": Tool("write_file", "Menulis teks ke sebuah file baru atau yang sudah ada.", self.file_manager.write_file, ["path", "content"]),
            
            # Alat dari LearningActuator
            "learn_from_file": Tool("learn_from_file", "Menganalisis, memahami, dan menyimpan pengetahuan dari file.", self.learning_actuator.learn_from_file, ["path"]),
            "synthesize_and_learn": Tool("synthesize_and_learn", "Berpikir tentang topik, menulis hasilnya, lalu mempelajarinya.", self.learning_actuator.synthesize_and_learn, ["topic", "new_file_path"]),
            
            # Alat dari ReasoningActuator (<- Alat baru)
            "deduce_from_sources": Tool("deduce_from_sources", "Menarik kesimpulan atau menalar tentang topik dari beberapa file sumber.", self.reasoning_actuator.deduce_from_sources, ["topic", "source_paths"]),
            "reason_and_synthesize": Tool("reason_and_synthesize", "Menalar dari file sumber, menyimpan kesimpulan ke file baru, dan mempelajarinya.", self.reasoning_actuator.reason_and_synthesize, ["topic", "source_paths", "new_file_path"])
        }

        for name, tool in all_tools.items():
            self.register_tool(tool)

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool
        self.tool_vectors[tool.name] = self.flock._text_to_vector(tool.description, self.flock.symbolic_engine.dim)

    def select_tool(self, stimulus_vector: List[float]) -> Optional[str]:
        # ... (implementasi tidak berubah)
        best_tool, max_similarity = None, -1.0
        for name, tool_vector in self.tool_vectors.items():
            similarity = self.flock._cosine_similarity(stimulus_vector, tool_vector)
            if similarity > max_similarity:
                max_similarity, best_tool = similarity, name
        return best_tool if max_similarity > 0.48 else None # Naikkan threshold sedikit lagi

    def execute_tool(self, tool_name: str, stimulus: str) -> str:
        # ... (implementasi tidak berubah, tapi sekarang bisa handle alat penalaran)
        tool = self.tools.get(tool_name)
        if not tool: return f"Error: Alat '{tool_name}' tidak ditemukan."
        try:
            args = self._extract_arguments(tool, stimulus)
            missing_params = [p for p in tool.parameters if p not in args]
            if missing_params:
                return f"Error: Parameter hilang ({missing_params}) untuk alat '{tool_name}'."
            result = tool.function(**args)
            return f"Hasil dari '{tool_name}': {result}"
        except Exception as e:
            return f"Error saat menjalankan '{tool_name}': {e}"

    def _extract_arguments(self, tool: Tool, stimulus: str) -> Dict[str, Any]:
        args = {}
        # Ekstraksi path (termasuk beberapa path)
        paths = re.findall(r"(?:file|sumber|dari|ke) ['"]?([\w/\.-]+)['"]?", stimulus, re.I)
        if paths:
            if "source_paths" in tool.parameters: args["source_paths"] = paths
            if "path" in tool.parameters: args["path"] = paths[0]
            if "new_file_path" in tool.parameters: args["new_file_path"] = paths[-1]

        # Ekstraksi topik
        topic_match = re.search(r"(?:topik|tentang) ['"](.*?)['"]", stimulus, re.I)
        if topic_match: args["topic"] = topic_match.group(1)

        # Ekstraksi konten
        content_match = re.search(r"(?:isi|konten|dengan) ['"](.*?)['"]", stimulus, re.I)
        if content_match: args["content"] = content_match.group(1)

        return args

@dataclass
class Thought: # ... (tidak berubah)
    content: str; vector: List[float]; confidence: float; timestamp: float = field(default_factory=time.time); source: str = "unknown"

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128, neural_topology: List[int] = [128, 64, 32], use_persistence: bool = True):
        # ... (Inisialisasi core, neuromodulators, etc. tidak berubah)
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim, attention_heads=4)
        self.neural_kernel = CognitiveKernel(neural_topology[0], neural_topology[1], neural_topology[2], symbolic_dim)
        self.chain_engine = ChainOfThought(self.symbolic_engine)
        self.neuromodulators = NeuromodulatorSystem()
        self.reflection = Reflection(self.neuromodulators)
        self.memory_store = MemoryStore() if use_persistence else None
        self.thought_history: List[Thought] = []
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        # ToolOrchestrator sekarang menginisialisasi semua alat
        self.tool_orchestrator = ToolOrchestrator(self)
    
    # ... (Semua fungsi utilitas: _text_to_vector, _cosine_similarity, dll. tidak berubah)
    def _text_to_vector(self, text: str, target_dim: int): # ...
        vec = [0.0] * target_dim
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = clean_text.split()
        if not tokens: return vec
        ngrams = list(tokens) + [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens) - 1)]
        for ngram in ngrams:
            hash1 = int.from_bytes(hashlib.md5(ngram.encode()).digest(), 'big')
            hash2 = int.from_bytes(hashlib.sha256(ngram.encode()).digest(), 'big')
            vec[hash1 % target_dim] += (hash2 % 2) * 2 - 1
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0: return [v/norm for v in vec]
        return vec

    def _cosine_similarity(self, v1, v2): return sum(a*b for a,b in zip(v1,v2))/(math.sqrt(sum(a*a for a in v1))*math.sqrt(sum(b*b for b in v2))+1e-9)

    def process_stimulus(self, stimulus: str) -> Dict[str, Any]:
        # ... (Logika tidak berubah, tapi sekarang lebih powerful karena alat yang lebih banyak)
        self.cycle_count += 1
        stim_symbolic = self._text_to_vector(stimulus, self.symbolic_engine.dim)
        selected_tool = self.tool_orchestrator.select_tool(stim_symbolic)
        
        if selected_tool:
            tool_result = self.tool_orchestrator.execute_tool(selected_tool, stimulus)
            thought = Thought(content=f"Hasil alat '{selected_tool}': {tool_result}", vector=self._text_to_vector(tool_result, self.symbolic_engine.dim), confidence=0.98, source="tool")
            self.thought_history.append(thought)
            return {"response_text": tool_result, "confidence": 0.98, "source": "tool", "cycle": self.cycle_count}
        
        return self._internal_reasoning(stimulus, stim_symbolic)

    def _internal_reasoning(self, stimulus: str, stim_symbolic: List[float]):
        # ... (Logika penalaran internal tidak berubah)
        stim_neural = self._text_to_vector(stimulus, 128)
        sym_res = self.symbolic_engine.query(stim_symbolic)
        cot_sum = self.chain_engine.reason(stimulus, steps=3)
        neu_res = self.neural_kernel.process(stim_neural)
        w_sym,w_neu,w_chain = 0.4,0.4,0.2 # simplified weights
        min_len = min(len(sym_res), len(neu_res), 32)
        combined_vec = [w_sym*sym_res[i] + w_neu*neu_res[i] + w_chain*(cot_sum[i] if i<len(cot_sum) else 0) for i in range(min_len)]
        agreement = self._compute_agreement(sym_res, neu_res, cot_sum)
        thought = Thought(content=self._vector_to_text(combined_vec), vector=combined_vec, confidence=agreement, source="ensemble")
        self.thought_history.append(thought)
        reflection_score = self.reflection.evaluate([t.vector for t in self.thought_history[-5:]])
        reward = agreement * 0.5 + reflection_score * 0.5
        self.cumulative_reward += reward
        if agreement < 0.3: self.neuromodulators.update_all(deltas=NeuromodulatoryEvent.novelty_detection(0.7))
        elif agreement > 0.8: self.neuromodulators.update_all(deltas=NeuromodulatoryEvent.reward_prediction_error(reward, 0.5))
        response_text = self._state_to_natural_language(combined_vec)
        return {"response_text": response_text, "response_vector": combined_vec, "confidence": agreement, "source": "ensemble"}

    def _vector_to_text(self, vec: List[float]): return f"vec[{vec[0]:.2f}, ...]"
    def _compute_agreement(self, v1,v2,v3): return(self._cosine_similarity(v1,v2)+self._cosine_similarity(v2,v3)+self._cosine_similarity(v3,v1))/3
    
    def deliberate(self, problem: str, max_cycles: int=5) -> str:
        # ... (implementasi tidak berubah)
        for i in range(max_cycles):
            result = self.process_stimulus(problem)
            if result.get("source") == "tool": return result["response_text"]
            if result["confidence"] > 0.8: break
            problem = result["response_text"]
        final_thought = self.thought_history[-1]
        if "vec[" in final_thought.content: return self._state_to_natural_language(final_thought.vector)
        return final_thought.content

    def _state_to_natural_language(self, state: List[float]) -> str: # ... (tidak berubah)
        top_indices = sorted(range(len(state)), key=lambda i: state[i], reverse=True)[:5]
        return f"Kesimpulan abstrak terfokus pada konsep: {top_indices}."
