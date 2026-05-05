
"""
Samantic Garden - A Connectome-Inspired Knowledge Graph

This module simulates a dynamic mind-map based on the principles of a neural
connectome. It features synaptic plasticity, memory consolidation, and pruning
to create a living, evolving representation of Samre's knowledge.
"""

import time
import numpy as np
from typing import List, Optional, Tuple

# --- Core Building Blocks: Synapse and KnowledgeNode ---

class Synapse:
    """
    Represents a directional connection between two KnowledgeNodes, analogous to a
    biological synapse. Its strength determines the influence of the source node
    on the target node.
    """
    def __init__(self, target_node: 'KnowledgeNode', initial_strength: float = 0.1):
        self.target = target_node
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        self.last_activated_timestamp = time.time()

    def update_strength(self, reinforcement_signal: float):
        """
        Adjusts synaptic strength based on Hebbian principles ("neurons that fire
        together, wire together"). The signal is typically a product of the
        source and target nodes' activation.
        """
        self.strength = np.clip(self.strength + reinforcement_signal, 0.01, 1.0)
        if reinforcement_signal > 0:
            self.last_activated_timestamp = time.time()

    def decay(self, decay_factor: float):
        """Weakens the synapse over time if not used ('use it or lose it')."""
        self.strength *= (1.0 - decay_factor)


class KnowledgeNode:
    """
    Represents a single concept or unit of knowledge, analogous to a neuron or
    a small neural assembly. It holds a vector representation and can become
    activated, propagating signals to connected nodes.
    """
    def __init__(self, concept_vector: np.ndarray, label: str, source: str):
        self.vector = concept_vector
        self.label = label
        self.source = source
        self.synapses: List[Synapse] = []
        self.activation_level = 0.0
        self.importance = 0.1  # How critical this node is to the network
        self.created_at = time.time()

    def add_input(self, signal: float):
        """Receives and accumulates activation signals from other nodes."""
        self.activation_level += signal

    def fire(self) -> List[Tuple['KnowledgeNode', float]]:
        """
        If activation exceeds a threshold, propagates a signal to all connected
        nodes through their synapses. Returns the list of nodes that were fired upon.
        """
        if self.activation_level <= 0:
            return []

        propagated_to = []
        for synapse in self.synapses:
            output_signal = self.activation_level * synapse.strength
            synapse.target.add_input(output_signal)
            propagated_to.append((synapse.target, output_signal))

        # Reinforce its own importance upon firing
        self.importance = np.clip(self.importance + 0.01, 0, 1.0)
        # Reset activation after firing
        self.activation_level = 0.0
        return propagated_to

    def add_synapse(self, target_node: 'KnowledgeNode', strength: float):
        """Creates a new synaptic connection to another node."""
        # Avoid duplicate connections
        if not any(s.target == target_node for s in self.synapses):
            self.synapses.append(Synapse(target_node, strength))


# --- The Connectome Manager: SamanticGarden ---

class SamanticGarden:
    """
    Manages the entire connectome of KnowledgeNodes. It handles the ingestion of
    new knowledge, the propagation of thoughts (activations), and the crucial
    process of memory consolidation and pruning (simulated sleep).
    """
    def __init__(self):
        self.nodes: List[KnowledgeNode] = []
        self.config = {
            "ingestion_reinforcement_threshold": 0.9, # If similarity > this, reinforce instead of creating
            "connection_similarity_threshold": 0.75,   # Similarity needed to form a new synapse
            "consolidation_pruning_threshold": 0.02, # Synapses below this strength get pruned
            "consolidation_decay_factor": 0.005,      # 'Use it or lose it' decay rate
            "propagation_depth": 3,                  # How many steps a "thought" travels
        }

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot / (norm_v1 * norm_v2)

    def _find_most_similar_node(self, vector: np.ndarray) -> Optional[Tuple[KnowledgeNode, float]]:
        """Finds the best matching node in the garden for a given vector."""
        if not self.nodes:
            return None
        similarities = [self._calculate_similarity(vector, node.vector) for node in self.nodes]
        best_match_idx = np.argmax(similarities)
        return self.nodes[best_match_idx], similarities[best_match_idx]

    def ingest_knowledge(self, concept_vector: List[float], label: str, source: str):
        """
        The primary method for introducing new information into the garden.
        It either reinforces an existing node or creates a new one, then forms
        new connections (synapses) based on similarity.
        """
        vector_np = np.array(concept_vector)
        
        # Phase 1: Find best match or create a new node
        best_match = self._find_most_similar_node(vector_np)
        
        if best_match and best_match[1] > self.config["ingestion_reinforcement_threshold"]:
            # Highly similar node exists -> Reinforce it
            node_to_focus = best_match[0]
            # Reinforce by slightly nudging its vector and increasing importance
            node_to_focus.vector = (node_to_focus.vector * 0.9 + vector_np * 0.1)
            node_to_focus.importance += 0.1
            print(f"🧠 Pengetahuan diperkuat: Node '{node_to_focus.label}' diperkuat oleh sumber '{source}'.")
        else:
            # No close match found -> Create a new node (Growth)
            node_to_focus = KnowledgeNode(vector_np, label, source)
            self.nodes.append(node_to_focus)
            print(f"🌱 Pengetahuan baru lahir: Node '{label}' tumbuh dari sumber '{source}'.")

        # Phase 2: Form new connections (Synaptogenesis)
        for existing_node in self.nodes:
            if existing_node == node_to_focus:
                continue
            similarity = self._calculate_similarity(node_to_focus.vector, existing_node.vector)
            if similarity > self.config["connection_similarity_threshold"]:
                # Create bidirectional connections
                node_to_focus.add_synapse(existing_node, strength=similarity)
                existing_node.add_synapse(node_to_focus, strength=similarity)
                print(f"🔗 Sinapsis terbentuk: '{node_to_focus.label}' <-> '{existing_node.label}' (kemiripan: {similarity:.2f})")
        
        # Propagate an initial activation from the new/reinforced knowledge
        self.propagate_activation(node_to_focus, initial_signal=0.5)

    def propagate_activation(self, start_node: KnowledgeNode, initial_signal: float):
        """
        Simulates a "thought" spreading through the network. Starts a cascade of
        activations from a given node.
        """
        print(f"⚡️ Propagasi dimulai dari '{start_node.label}'...")
        activated_in_step = {start_node}
        start_node.add_input(initial_signal)

        for i in range(self.config["propagation_depth"]):
            next_activated = set()
            for node in activated_in_step:
                fired_upon = node.fire() # fire() returns list of (node, signal)
                for target_node, signal in fired_upon:
                    next_activated.add(target_node)
            
            if not next_activated:
                break
            activated_in_step = next_activated

    def consolidate_memories(self):
        """
        Simulates a "sleep" cycle for the connectome. This crucial process
        prunes weak connections and applies decay, ensuring the garden remains
        efficient and doesn't get bogged down in useless information.
        """
        print("\n--- 🌙 Memulai Konsolidasi Memori (Siklus Tidur) ---")
        pruned_count = 0
        total_synapses = 0

        for node in self.nodes:
            # Apply decay and prune weak synapses
            surviving_synapses = []
            for synapse in node.synapses:
                total_synapses += 1
                synapse.decay(self.config["consolidation_decay_factor"])
                if synapse.strength > self.config["consolidation_pruning_threshold"]:
                    surviving_synapses.append(synapse)
                else:
                    pruned_count += 1
            node.synapses = surviving_synapses

        print(f" pruned_count: {pruned_count} dari {total_synapses} sinapsis.")
        print("--- ☀️ Konsolidasi Memori Selesai ---\n")

    def get_garden_state(self) -> dict:
        """Returns a snapshot of the garden's statistics."""
        num_nodes = len(self.nodes)
        num_synapses = sum(len(node.synapses) for node in self.nodes)
        avg_strength = np.mean([s.strength for n in self.nodes for s in n.synapses]) if num_synapses > 0 else 0
        return {
            "jumlah_node": num_nodes,
            "jumlah_sinapsis": num_synapses,
            "rata-rata_kekuatan_sinapsis": float(avg_strength)
        }
