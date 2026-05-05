"""
Flock of Thought - Inti Kognitif Otonom dari Samre.

Kelas ini mengatur seluruh siklus otonom, mulai dari memperbarui kebutuhan
hingga melaksanakan tindakan dan belajar dari hasilnya, mendelegasikan tugas
ke aktuator khusus.
"""
import time
import numpy as np
from typing import Dict, Any, List

# Komponen Inti
from core.cognitive_core import CognitiveEngine
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action
from tools.file_manager import FileManager
from core.actuators import LearningActuator, ReasoningActuator, EvolutionaryActuator
from core.samantic_garden import KnowledgeNode, Synapse

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128):
        print("🧠 Memulai Flock of Thought (Arsitektur Lengkap)...")
        # Arsitektur Kognitif & Kebutuhan
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim)
        self.neuromodulators = NeuromodulatorSystem()
        self.needs = InternalNeeds()

        # Komponen Eksekusi & Pengetahuan
        self.file_manager = FileManager()
        self.knowledge_base: Dict[str, Any] = {}
        self.learning_actuator = LearningActuator(self.file_manager, self.knowledge_base)
        self.reasoning_actuator = ReasoningActuator(self.knowledge_base)
        self.evolutionary_actuator = EvolutionaryActuator(self.file_manager)

        # Memori Jangka Panjang untuk Kinerja Tindakan
        self.ltm_action_success = {action: {"success": 0, "total": 0} for action in POSSIBLE_ACTIONS}

        # Taman Samantik untuk representasi pengetahuan
        self.samantic_garden: List[KnowledgeNode] = []
        self.similarity_threshold = 0.8  # Ambang batas untuk membuat sinapsis

        # Status Siklus
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        print("✅ Flock of Thought telah diinisialisasi dengan semua komponen.")

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Menghitung cosine similarity antara dua vektor."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _create_and_link_node(self, concept_vector: List[float], label: str, source: str):
        """Membuat Node Pengetahuan baru dan menghubungkannya ke node yang ada di taman."""
        new_node = KnowledgeNode(concept_vector, label=label, source_file=source)
        print(f"🌱 Menumbuhkan Node Pengetahuan baru: '{label}' dari sumber '{source}'")

        # Hubungkan ke node yang ada berdasarkan kesamaan
        for existing_node in self.samantic_garden:
            similarity = self._calculate_cosine_similarity(new_node.vector, existing_node.vector)
            if similarity > self.similarity_threshold:
                # Buat koneksi dua arah
                new_node.synapses.append(Synapse(existing_node, initial_strength=similarity))
                existing_node.synapses.append(Synapse(new_node, initial_strength=similarity))
                print(f"🔗 Menghubungkan node '{new_node.label}' dengan '{existing_node.label}' (kesamaan: {similarity:.2f})")
        
        self.samantic_garden.append(new_node)

    def update_and_score_actions(self) -> Dict[str, float]:
        """Langkah 1 & 2: Perbarui kebutuhan dan beri skor semua tindakan yang mungkin."""
        print("\n--- AWAL SIKLUS ---")
        self.cycle_count += 1
        print(f"🔄 Siklus: {self.cycle_count}")

        self.needs.update_needs()
        print(f"Kebutuhan diperbarui: {self.needs.get_all_needs()}")

        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            self.neuromodulators.get_all_levels()
        )
        print(f"Skor tindakan: {action_scores}")
        return action_scores

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        """Langkah 3: Evaluasi tindakan yang diberi skor dan pilih satu."""
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)

        selected_action = "REST"
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), self.neuromodulators.get_all_levels()):
                selected_action = action
                break
        
        print(f"Tindakan dievaluasi & dipilih: {selected_action}")
        return selected_action

    def execute_action(self, action: str) -> None:
        """Langkah 4, 5, 6: Jalankan, catat imbalan, dan perbarui pembelajaran."""
        print(f"⚡️ Mendelegasikan eksekusi untuk tindakan: {action}")
        reward = 0.0
        execution_success = False

        try:
            if action == "LEARN":
                # Di masa depan, aktuator akan mengembalikan data yang dipelajari
                execution_success = self.learning_actuator.execute()
                if execution_success:
                    self.needs.satisfy_need("hunger", 0.7)
                    reward = 0.8
                    # Placeholder: Buat node baru di Samantic Garden
                    # Vektor konsep dan label akan berasal dari hasil aktuator
                    last_source = self.learning_actuator.last_source or "unknown"
                    new_concept_vector = self.symbolic_engine.query(np.random.rand(self.symbolic_engine.dim).tolist())
                    self._create_and_link_node(
                        concept_vector=new_concept_vector,
                        label=f"LearnedConcept_{len(self.samantic_garden)}",
                        source=last_source
                    )
                else:
                    reward = -0.5

            elif action == "REASON":
                execution_success = self.reasoning_actuator.execute()
                if execution_success:
                    reward = 0.6
                else:
                    reward = -0.4
                self.needs.satisfy_need("hunger", 0.2)

            elif action == "EVOLVE":
                execution_success = self.evolutionary_actuator.execute()
                if execution_success:
                    reward = 0.3 # Imbalan awal yang kecil untuk analisis yang berhasil
                else:
                    reward = -0.8
                self.needs.satisfy_need("messiness", 0.5)

            elif action == "REST":
                print("💤 REST: Beristirahat untuk memulihkan kelelahan.")
                time.sleep(2)
                self.needs.satisfy_need("fatigue", 0.9)
                reward = 0.2
                execution_success = True
            
            else:
                print(f"Tindakan '{action}' belum memiliki aktuator khusus. Melewati.")
                time.sleep(1)
                reward = 0.1
                execution_success = True # Placeholder dianggap berhasil

        except Exception as e:
            print(f"💥 KESALAHAN KRITIS saat menjalankan {action}: {e}")
            reward = -1.0 # Penalti berat untuk kesalahan tak terduga
            execution_success = False

        # Langkah 5: Catat Imbalan
        self.cumulative_reward += reward
        print(f"📊 Hasil: Berhasil={execution_success}, Imbalan={reward:.2f}, Imbalan Kumulatif={self.cumulative_reward:.3f}")

        # Langkah 6: Perbarui Sistem Pembelajaran
        self.update_learning_systems(action, execution_success, reward)

    def update_learning_systems(self, action: str, success: bool, reward: float):
        """Memperbarui memori jangka panjang dan sistem neuromodulatori."""
        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1
        
        # Hitung Reward Prediction Error (RPE) sederhana
        expected_reward = 0.1
        rpe = reward - expected_reward
        
        # Perbarui neuromodulator berdasarkan RPE dan peristiwa lain
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, expected_reward)
        self.neuromodulators.update_all(deltas)
        print(f"Neuromodulator diperbarui: {self.neuromodulators.get_all_levels()}")
        print("--- AKHIR SIKLUS ---")
