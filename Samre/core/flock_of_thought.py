"""
Flock of Thought - The Autonomous Cognitive Core of Samre.

This class orchestrates the entire autonomous cycle, from updating needs to
executing actions and learning from the outcomes, delegating tasks to
-specialized actuators.
"""
import time
import os
import numpy as np
from typing import Dict, List, Set
from collections import deque

# Core Components
from core.cognitive_core import CognitiveEngine
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action

# Tools & Actuators
from tools.file_manager import FileManager
from tools.binary_reader import BinaryReader
from tools.text_processor import TextProcessor # <-- IMPORT PATH UPDATED
from core.actuators import ExploreActuator, LearningActuator, ReasoningActuator, EvolutionaryActuator
from core.samantic_garden import SamanticGarden

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128):
        print("🧠 Initializing Flock of Thought (Full Architecture)...")
        # Cognitive & Need Architecture
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim)
        self.neuromodulators = NeuromodulatorSystem()
        self.needs = InternalNeeds()

        # Tools & Actuators
        self.file_manager = FileManager()
        self.text_processor = TextProcessor(vector_dim=symbolic_dim)
        self.binary_reader = BinaryReader()
        self.explore_actuator = ExploreActuator(self.file_manager)
        self.learning_actuator = LearningActuator(self.file_manager)
        self.reasoning_actuator = ReasoningActuator()
        self.evolutionary_actuator = EvolutionaryActuator(self.file_manager)
        
        # The Connectome-Inspired Mind-Map
        self.samantic_garden = SamanticGarden()

        # State & Memory
        self.ltm_action_success = {action: {"success": 0, "total": 0} for action in POSSIBLE_ACTIONS}
        self.explored_paths: Set[str] = set()
        self.learning_queue: deque[str] = deque()
        self.text_file_extensions = {'.py', '.md', '.txt', '.json', '.xml', '.html', '.css', '.js'}

        # Cycle Status & Bootstrap
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        self._bootstrap_initial_knowledge()
        print("✅ Flock of Thought initialized and bootstrapped.")

    def _bootstrap_initial_knowledge(self):
        """Read own source code to build an initial IDF model."""
        print("--- Bootstrapping Self-Awareness ---")
        # Correctly defined list of core source files for bootstrapping
        source_files = [
            "core/flock_of_thought.py", "core/cognitive_core.py",
            "core/samantic_garden.py", "core/needs.py", "core/sws_logic.py",
            "core/actuators.py", "core/executive.py", "core/neuromodulator.py",
            "tools/text_processor.py", # <-- BOOTSTRAP PATH UPDATED
            "tools/file_manager.py", "tools/binary_reader.py",
            "main.py",
        ]
        corpus = []
        for file_path in source_files:
            # Ensure the agent doesn't try to re-learn its own code right away
            self.explored_paths.add(file_path)
            read_result = self.file_manager.read_file(file_path)
            if "content" in read_result:
                corpus.append(read_result["content"])
            else:
                print(f"⚠️  Could not read own file for bootstrap: {file_path}")
        
        if corpus:
            self.text_processor.update_idf_counts(corpus)
        print("--- Bootstrap Complete ---")

    def update_and_score_actions(self) -> Dict[str, float]:
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")
        self.needs.update_needs()
        print(f"Needs Updated: {self.needs.get_all_needs()}")
        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            self.neuromodulators.get_all_levels()
        )
        print(f"Action Scores: {action_scores}")
        return action_scores

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        selected_action = "REST"
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), self.neuromodulators.get_all_levels()):
                selected_action = action
                break
        print(f"Action Evaluated & Selected: {selected_action}")
        return selected_action

    def execute_action(self, action: str) -> None:
        print(f"⚡️ Delegating execution for action: {action}")
        reward = 0.0
        execution_success = False
        try:
            if action == "EXPLORE":
                new_file_path = self.explore_actuator.execute(self.explored_paths)
                if new_file_path:
                    self.learning_queue.append(new_file_path)
                    self.explored_paths.add(new_file_path)
                    reward = 0.5
                    execution_success = True
                else:
                    reward = -0.1
                    execution_success = False

            elif action == "LEARN":
                if not self.learning_queue:
                    print("🔎 LEARNING queue is empty. Nothing to learn.")
                    reward = -0.3
                    execution_success = False
                else:
                    file_to_learn = self.learning_queue.popleft()
                    file_ext = os.path.splitext(file_to_learn)[1].lower()
                    content_to_process = None
                    source_label = ""

                    if file_ext in self.text_file_extensions:
                        source_label = f"Text file: {file_to_learn}"
                        learn_result = self.learning_actuator.execute_text(file_to_learn)
                        if learn_result:
                            content_to_process = learn_result[0]
                    else:
                        source_label = f"Binary file: {file_to_learn}"
                        print(f"🔬 LEARNING (Binary): Probing '{file_to_learn}'.")
                        binary_data = self.binary_reader.read_binary_file(os.path.join(self.file_manager.base_path, file_to_learn))
                        if isinstance(binary_data, bytes):
                            extracted_strings = self.binary_reader.extract_printable_strings(binary_data)
                            if extracted_strings:
                                content_to_process = " ".join(extracted_strings)
                                print(f"    ✅ Extracted {len(extracted_strings)} strings from binary.")
                            else:
                                print("    ⚠️ No printable strings found in binary.")
                        else:
                             print(f"    ❌ {binary_data.get('error')}")
                    
                    if content_to_process:
                        vector, keywords = self.text_processor.text_to_vector(content_to_process)
                        concept_vector = self.symbolic_engine.query(vector)
                        label = f"Concept: {', '.join(keywords.keys())}"
                        self.samantic_garden.ingest_knowledge(concept_vector, label, source_label, list(keywords.keys()))
                        self.needs.satisfy_need("hunger", 0.8)
                        self.needs.increase_need("cognitive_load", 0.4)
                        reward = 0.9
                        execution_success = True
                    else:
                        reward = -0.5
                        execution_success = False

            elif action == "CONSOLIDATE":
                self.samantic_garden.consolidate_memories()
                self.needs.satisfy_need("cognitive_load", 0.9)
                reward = 0.7
                execution_success = True

            elif action == "REASON":
                execution_success = self.reasoning_actuator.execute()
                if execution_success:
                    reward = 0.4
                else:
                    reward = -0.2

            elif action == "EVOLVE":
                execution_success = self.evolutionary_actuator.execute()
                if execution_success:
                    reward = 0.3
                else:
                    reward = -0.6

            elif action == "REST":
                time.sleep(1)
                self.needs.satisfy_need("fatigue", 0.9)
                reward = 0.2
                execution_success = True
            
            else:
                print(f"Action '{action}' is not fully implemented yet. Skipping.")
                reward = 0.05
                execution_success = True

        except Exception as e:
            print(f"💥 CRITICAL ERROR during {action} execution: {e}")
            reward = -1.0
            execution_success = False

        self.cumulative_reward += reward
        print(f"📊 Outcome: Success={execution_success}, Reward={reward:.2f}, Cumulative Reward={self.cumulative_reward:.3f}")
        self.update_learning_systems(action, execution_success, reward)

    def update_learning_systems(self, action: str, success: bool, reward: float):
        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1
        
        expected_reward = 0.1
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, expected_reward)
        self.neuromodulators.update_all(deltas=deltas)
        print(f"Neuromodulators Updated: {self.neuromodulators.get_all_levels()}")
        print(f"Garden State: {self.samantic_garden.get_garden_state()}")
        print("--- CYCLE END ---")
