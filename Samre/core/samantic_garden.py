# core/samantic_garden.py (Representasi Konseptual)
import time
from typing import List

class Synapse:
    """Koneksi antara dua KnowledgeNode."""
    def __init__(self, target_node, initial_strength=0.1):
        self.target = target_node
        self.strength = initial_strength  # Kekuatan koneksi (0-1)
        self.last_activated = time.time()
        self.activation_count = 0

class KnowledgeNode:
    """Satu unit pengetahuan dalam Mind-Mapping."""
    def __init__(self, concept_vector, label=None, source_file=None):
        self.vector = concept_vector  # Representasi vektor dari konsep ini
        self.label = label  # Nama atau deskripsi (opsional)
        self.source = source_file  # Dari mana pengetahuan ini berasal
        self.synapses: List[Synapse] = []  # Koneksi ke node lain
        self.importance = 0.1  # Seberapa penting node ini
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def fire(self, activation_signal: float):
        """Aktivasi node ini dan propagasi ke tetangga."""
        self.last_accessed = time.time()
        self.importance += 0.01 * activation_signal
        # Propagasi ke semua sinapsis
        propagated = []
        for syn in self.synapses:
            output = activation_signal * syn.strength
            syn.activation_count += 1
            syn.last_activated = time.time()
            syn.strength = min(1.0, syn.strength + 0.001)  # Hebbian reinforcement
            propagated.append((syn.target, output))
        return propagated
