"""
Samantic Garden - A Connectome-Inspired Knowledge Graph

This module simulates a dynamic mind-map based on the principles of a neural
connectome. It features synaptic plasticity, memory consolidation, and pruning
to create a living, evolving representation of Samre's knowledge.
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Dict
import os
from datetime import datetime

# --- Visualization Imports (with fallback) ---
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    print("⚠️  Matplotlib or NetworkX not found. Visualization is disabled.")

# --- Core Building Blocks: Synapse and KnowledgeNode ---

class Synapse:
    """
    Represents a directional connection between two KnowledgeNodes.
    """
    def __init__(self, target_node: 'KnowledgeNode', initial_strength: float = 0.1):
        self.target = target_node
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        self.last_activated_timestamp = time.time()

    def update_strength(self, reinforcement_signal: float):
        self.strength = np.clip(self.strength + reinforcement_signal, 0.01, 1.0)
        if reinforcement_signal > 0:
            self.last_activated_timestamp = time.time()

    def decay(self, decay_factor: float):
        self.strength *= (1.0 - decay_factor)


class KnowledgeNode:
    """
    Represents a single concept or unit of knowledge.
    """
    def __init__(self, concept_vector: np.ndarray, label: str, source: str, keywords: List[str] = []):
        self.id = f"{label.replace(' ', '')}_{int(time.time()*1000)}" # Unique ID
        self.vector = concept_vector
        self.label = label
        self.source = source
        self.keywords = keywords
        self.synapses: List[Synapse] = []
        self.activation_level = 0.0
        self.importance = 0.1
        self.created_at = time.time()

    def add_input(self, signal: float):
        self.activation_level += signal

    def fire(self) -> List[Tuple['KnowledgeNode', float]]:
        if self.activation_level <= 0:
            return []
        propagated_to = []
        for synapse in self.synapses:
            output_signal = self.activation_level * synapse.strength
            synapse.target.add_input(output_signal)
            propagated_to.append((synapse.target, output_signal))
        self.importance = np.clip(self.importance + 0.01, 0, 1.0)
        self.activation_level = 0.0
        return propagated_to

    def add_synapse(self, target_node: 'KnowledgeNode', strength: float):
        if not any(s.target == target_node for s in self.synapses):
            self.synapses.append(Synapse(target_node, strength))


# --- The Connectome Manager: SamanticGarden ---

class SamanticGarden:
    """
    Manages the entire connectome of KnowledgeNodes, including ingestion,
    activation propagation, consolidation, and visualization.
    """
    def __init__(self, log_dir="Samre/log"):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.log_dir = log_dir
        self.config = {
            "ingestion_reinforcement_threshold": 0.9,
            "connection_similarity_threshold": 0.75,
            "consolidation_pruning_threshold": 0.02,
            "consolidation_decay_factor": 0.005,
            "propagation_depth": 3,
        }
        if VISUALIZATION_ENABLED and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"Log directory created at: {self.log_dir}")

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # (Identical to previous version)
        dot = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot / (norm_v1 * norm_v2)

    def _find_most_similar_node(self, vector: np.ndarray) -> Optional[Tuple[KnowledgeNode, float]]:
        if not self.nodes:
            return None
        node_list = list(self.nodes.values())
        similarities = [self._calculate_similarity(vector, node.vector) for node in node_list]
        best_match_idx = np.argmax(similarities)
        return node_list[best_match_idx], similarities[best_match_idx]

    def ingest_knowledge(self, concept_vector: List[float], label: str, source: str, keywords: List[str]):
        vector_np = np.array(concept_vector)
        best_match = self._find_most_similar_node(vector_np)
        
        if best_match and best_match[1] > self.config["ingestion_reinforcement_threshold"]:
            node_to_focus = best_match[0]
            node_to_focus.vector = (node_to_focus.vector * 0.9 + vector_np * 0.1)
            node_to_focus.importance += 0.1
            print(f"🧠 Knowledge Reinforced: Node '{node_to_focus.label}' was strengthened by source '{source}'.")
        else:
            node_to_focus = KnowledgeNode(vector_np, label, source, keywords)
            self.nodes[node_to_focus.id] = node_to_focus
            print(f"🌱 New Knowledge Born: Node '{label}' grew from source '{source}'.")

        for existing_node in self.nodes.values():
            if existing_node == node_to_focus:
                continue
            similarity = self._calculate_similarity(node_to_focus.vector, existing_node.vector)
            if similarity > self.config["connection_similarity_threshold"]:
                node_to_focus.add_synapse(existing_node, strength=similarity)
                existing_node.add_synapse(node_to_focus, strength=similarity)
                print(f"🔗 Synapse Formed: '{node_to_focus.label}' <-> '{existing_node.label}' (similarity: {similarity:.2f})")
        
        self.propagate_activation(node_to_focus, initial_signal=0.5)

    def propagate_activation(self, start_node: KnowledgeNode, initial_signal: float):
        # (Identical to previous version)
        print(f"⚡️ Propagation started from '{start_node.label}'...")
        activated_in_step = {start_node}
        start_node.add_input(initial_signal)
        for _ in range(self.config["propagation_depth"]):
            next_activated = set()
            for node in activated_in_step:
                fired_upon = node.fire()
                for target_node, signal in fired_upon:
                    next_activated.add(target_node)
            if not next_activated:
                break
            activated_in_step = next_activated

    def consolidate_memories(self):
        print("\n--- 🌙 Initiating Memory Consolidation (Sleep Cycle) ---")
        pruned_count = 0
        total_synapses = sum(len(node.synapses) for node in self.nodes.values())
        for node in self.nodes.values():
            surviving_synapses = []
            for synapse in node.synapses:
                synapse.decay(self.config["consolidation_decay_factor"])
                if synapse.strength > self.config["consolidation_pruning_threshold"]:
                    surviving_synapses.append(synapse)
                else:
                    pruned_count += 1
            node.synapses = surviving_synapses
        print(f" pruned synapses: {pruned_count} out of {total_synapses}.")
        if VISUALIZATION_ENABLED:
            self.visualize_and_save_graph()
        print("--- ☀️ Memory Consolidation Complete ---\n")

    def visualize_and_save_graph(self):
        """Creates and saves a visual representation of the knowledge graph."""
        if not VISUALIZATION_ENABLED:
            return

        G = nx.Graph()
        node_labels = {}
        node_sizes = []
        
        for node in self.nodes.values():
            G.add_node(node.id)
            # Make labels shorter for readability
            clean_label = node.label.replace("Concept: ", "").split(',')[0]
            node_labels[node.id] = clean_label
            node_sizes.append(1000 + (node.importance * 4000)) # Size based on importance

        edge_widths = []
        for node in self.nodes.values():
            for synapse in node.synapses:
                if G.has_edge(node.id, synapse.target.id):
                    continue # Avoid duplicate edges in visualization
                G.add_edge(node.id, synapse.target.id, weight=synapse.strength)
                edge_widths.append(synapse.strength * 5)

        if not self.nodes:
            print("📊 Garden is empty. Nothing to visualize.")
            return

        print("📊 Generating graph visualization...")
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=0.9, iterations=50) # Spacious layout
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#cccccc', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

        plt.title("Samre's Samantic Garden", fontsize=20)
        plt.axis('off')
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.log_dir, f"samantic_garden_{timestamp}.png")
        try:
            plt.savefig(filename)
            print(f"✅ Graph visualization saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save graph: {e}")
        plt.close()

    def get_garden_state(self) -> dict:
        num_nodes = len(self.nodes)
        num_synapses = sum(len(node.synapses) for node in self.nodes.values())
        avg_strength = np.mean([s.strength for n in self.nodes.values() for s in n.synapses]) if num_synapses > 0 else 0
        return {
            "jumlah_node": num_nodes,
            "jumlah_sinapsis": num_synapses,
            "rata-rata_kekuatan_sinapsis": float(avg_strength)
        }
