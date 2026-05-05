"""
Flock of Thought - The Autonomous Cognitive Core of Samre.

This class orchestrates the entire autonomous cycle, from updating needs to
executing actions and learning from the outcomes, delegating tasks to
-specialized actuators.
"""
import time
import numpy as np
from typing import Dict, Any, List

# Core Components
from core.cognitive_core import CognitiveEngine
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action
from tools.file_manager import FileManager
from core.actuators import LearningActuator, ReasoningActuator, EvolutionaryActuator
from core.samantic_garden import SamanticGarden # Import the new connectome

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128):
        print("🧠 Initializing Flock of Thought (Full Architecture)...")
        # Cognitive & Need Architecture
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim)
        self.neuromodulators = NeuromodulatorSystem()
        self.needs = InternalNeeds()

        # Execution & Knowledge Components
        self.file_manager = FileManager()
        self.knowledge_base: Dict[str, Any] = {}
        self.learning_actuator = LearningActuator(self.file_manager, self.knowledge_base)
        self.reasoning_actuator = ReasoningActuator(self.knowledge_base)
        self.evolutionary_actuator = EvolutionaryActuator(self.file_manager)
        
        # The Connectome-Inspired Mind-Map
        self.samantic_garden = SamanticGarden()

        # Long-Term Memory for Action Performance
        self.ltm_action_success = {action: {"success": 0, "total": 0} for action in POSSIBLE_ACTIONS}

        # Cycle Status
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        print("✅ Flock of Thought initialized with all components.")

    def update_and_score_actions(self) -> Dict[str, float]:
        """Steps 1 & 2: Update needs and score all possible actions."""
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")

        self.needs.update_needs()
        print(f"Needs Updated: {self.needs.get_all_needs()}")

        # Pass LTM data to the scoring function
        ltm_data = {action: info['success'] / info['total'] if info['total'] > 0 else 0.5 
                    for action, info in self.ltm_action_success.items()}

        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            self.neuromodulators.get_all_levels()
        )
        print(f"Action Scores: {action_scores}")
        return action_scores

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        """Step 3: Evaluate scored actions and select one."""
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)

        selected_action = "REST"
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), self.neuromodulators.get_all_levels()):
                selected_action = action
                break
        
        print(f"Action Evaluated & Selected: {selected_action}")
        return selected_action

    def execute_action(self, action: str) -> None:
        """Steps 4, 5, 6: Execute, record reward, and update learning."""
        print(f"⚡️ Delegating execution for action: {action}")
        reward = 0.0
        execution_success = False

        try:
            if action == "LEARN":
                execution_success = self.learning_actuator.execute()
                if execution_success:
                    self.needs.satisfy_need("hunger", 0.7)
                    reward = 0.8
                    # Ingest the new knowledge into the Samantic Garden
                    last_source = self.learning_actuator.last_source or "unknown_source"
                    # Create a conceptual vector from the learned content (placeholder)
                    new_concept_vector = self.symbolic_engine.query(np.random.rand(self.symbolic_engine.dim).tolist())
                    self.samantic_garden.ingest_knowledge(
                        concept_vector=new_concept_vector,
                        label=f"LearnedConcept_{len(self.samantic_garden.nodes)}",
                        source=last_source
                    )
                    # Ingesting knowledge increases cognitive load
                    self.needs.increase_need("cognitive_load", 0.2)
                else:
                    reward = -0.5

            elif action == "CONSOLIDATE":
                print("🧠 CONSOLIDATE: Initiating memory consolidation...")
                self.samantic_garden.consolidate_memories()
                self.needs.satisfy_need("cognitive_load", 0.9)
                reward = 0.7 # High reward for maintaining cognitive health
                execution_success = True

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
                    reward = 0.3
                else:
                    reward = -0.8
                self.needs.satisfy_need("messiness", 0.5)

            elif action == "REST":
                print("💤 REST: Resting to recover from fatigue.")
                time.sleep(2)
                self.needs.satisfy_need("fatigue", 0.9)
                reward = 0.2
                execution_success = True
            
            else: # EXPLORE, ORGANIZE, etc.
                print(f"Action '{action}' does not have a dedicated actuator yet. Skipping.")
                time.sleep(1)
                reward = 0.1 # Small reward for trying something
                execution_success = True # Placeholder success

        except Exception as e:
            print(f"💥 CRITICAL ERROR during {action} execution: {e}")
            reward = -1.0 # Heavy penalty for unexpected failure
            execution_success = False

        # Step 5: Record Reward
        self.cumulative_reward += reward
        print(f"📊 Outcome: Success={execution_success}, Reward={reward:.2f}, Cumulative Reward={self.cumulative_reward:.3f}")

        # Step 6: Update Learning Systems
        self.update_learning_systems(action, execution_success, reward)

    def update_learning_systems(self, action: str, success: bool, reward: float):
        """Updates long-term memory and the neuromodulatory system."""
        # Update LTM
        self.ltm_action_success[action]["total"] += 1
        if success:
            self.ltm_action_success[action]["success"] += 1
        
        # Calculate simple Reward Prediction Error (RPE)
        # Assumed baseline reward is 0.1 (for a neutral action)
        expected_reward = 0.1
        rpe = reward - expected_reward
        
        # Update neuromodulators based on the RPE and other events
        deltas = NeuromodulatoryEvent.reward_prediction_error(reward, expected_reward)
        self.neuromodulators.update_all(deltas=deltas)
        print(f"Neuromodulators Updated: {self.neuromodulators.get_all_levels()}")
        print(f"Garden State: {self.samantic_garden.get_garden_state()}")
        print("--- CYCLE END ---")
