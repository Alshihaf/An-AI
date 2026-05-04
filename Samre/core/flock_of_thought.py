"""
Flock of Thought - The autonomous cognitive core of Samre.

This class orchestrates the entire autonomous loop, from updating needs to
executing actions and learning from the outcomes.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable, Tuple
import time
import re
import hashlib
import math
import json

# Core components
from core.cognitive_core import CognitiveEngine
from core.neural_ecosystem import CognitiveKernel, AutonomousANN
from core.neuromodulator import NeuromodulatorSystem, NeuromodulatoryEvent
from core.chain_of_thought import ChainOfThought
from core.reflection import Reflection
from core.needs import InternalNeeds
from core.sws_logic import score_all_actions, POSSIBLE_ACTIONS
from core.executive import evaluate_action

# Actuators and Memory
from memory.storage import MemoryStore
from tools.file_manager import FileManager
from act.learn import LearningActuator
from act.reason import ReasoningActuator

@dataclass
class Thought: 
    content: str
    vector: List[float]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

class FlockOfThought:
    def __init__(self, symbolic_dim: int = 128, use_persistence: bool = True):
        print("🧠 Initializing Flock of Thought...")
        # Core Cognitive Architecture
        self.symbolic_engine = CognitiveEngine(dimensionality=symbolic_dim)
        self.neuromodulators = NeuromodulatorSystem()
        self.needs = InternalNeeds()
        self.reflection = Reflection(self.neuromodulators)
        self.chain_engine = ChainOfThought(self.symbolic_engine)

        # Memory Systems
        self.memory_store = MemoryStore() if use_persistence else None
        self.ltm_action_success = {action: {"success": 0, "total": 0} for action in POSSIBLE_ACTIONS}

        # State & Cycle Management
        self.thought_history: List[Thought] = []
        self.cycle_count = 0
        self.cumulative_reward = 0.0

        # Actuators & Tools
        self.file_manager = FileManager()
        self.learning_actuator = LearningActuator(self, self.file_manager)
        self.reasoning_actuator = ReasoningActuator(self, self.file_manager, self.learning_actuator)
        print("✅ Flock of Thought initialized.")

    def update_and_score_actions(self) -> Dict[str, float]:
        """Step 1 & 2: Update needs and score all possible actions."""
        print("\n--- CYCLE START ---")
        self.cycle_count += 1
        print(f"🔄 Cycle: {self.cycle_count}")
        
        # 1. UPDATE_NEEDS
        self.needs.update_needs()
        print(f"Needs updated: {self.needs.get_all_needs()}")

        # 2. SCORE_ACTIONS
        action_scores = score_all_actions(
            self.needs.get_all_needs(),
            self.neuromodulators.get_all_levels()
            # LTM/CoT data would be passed here in a future version
        )
        print(f"Action scores: {action_scores}")
        return action_scores

    def evaluate_and_select_action(self, action_scores: Dict[str, float]) -> str:
        """Step 3: Evaluate scored actions and select one."""
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 3. EVALUATE_ACTIONS
        selected_action = "REST" # Default action
        for action, score in sorted_actions:
            if evaluate_action(action, score, self.needs.get_all_needs(), self.neuromodulators.get_all_levels()):
                selected_action = action
                break # Stop at the first approved action
        
        print(f"Action selected: {selected_action}")
        return selected_action

    def execute_action(self, action: str) -> None:
        """Step 4, 5, 6: Execute, record reward, and update learning."""
        print(f"Executing action: {action}...")
        reward = 0.0
        
        # 4. EXECUTE_ACTION
        try:
            # Simple execution logic for now.
            # This will become much more complex.
            if action == "EXPLORE":
                # Placeholder: Just list files in the current directory
                files = self.file_manager.list_dir(".")
                print(f"EXPLORE result: Found {files}")
                self.needs.satisfy_need("hunger", 0.5)
                self.needs.satisfy_need("boredom", 0.4)
                reward = 0.6
            elif action == "EVOLVE":
                # Placeholder: Simulate a code change
                print("EVOLVE: Analyzing codebase for potential improvements...")
                time.sleep(2) # Simulate work
                print("EVOLVE: Applied a minor refactoring (simulation).")
                self.needs.satisfy_need("messiness", 0.8)
                reward = 0.9
            elif action == "ORGANIZE":
                # Placeholder: Consolidate memory
                self.symbolic_engine.think(cycles=15) # Deepen thought
                print("ORGANIZE: Memory consolidated.")
                self.needs.satisfy_need("messiness", 0.4)
                reward = 0.5
            elif action == "LEARN":
                # Placeholder: Learn from the README file
                self.learning_actuator.learn_from_file("README.md")
                print("LEARN: Processed README.md")
                self.needs.satisfy_need("hunger", 0.6)
                reward = 0.7
            elif action == "REASON":
                # Placeholder: Reason about the core files
                self.reasoning_actuator.deduce_from_sources(
                    topic="system architecture", 
                    source_paths=["Samre/core/cognitive_core.py", "Samre/core/flock_of_thought.py"]
                )
                print("REASON: Deduced conclusions about system architecture.")
                self.needs.satisfy_need("hunger", 0.3)
                reward = 0.8
            elif action == "REST":
                time.sleep(3) # Simulate resting
                self.needs.satisfy_need("fatigue", 0.9)
                reward = 0.3
            
            execution_success = True
        except Exception as e:
            print(f"ERROR during {action}: {e}")
            execution_success = False
            reward = -0.5 # Negative reward on failure

        # 5. RECORD_REWARD
        self.cumulative_reward += reward
        print(f"Action Result: Success={execution_success}, Reward={reward}, Cumulative Reward={self.cumulative_reward:.3f}")

        # 6. UPDATE_LEARNING_SYSTEMS
        # a. Update Long-Term Memory of action success
        self.ltm_action_success[action]["total"] += 1
        if execution_success:
            self.ltm_action_success[action]["success"] += 1
        
        # b. Update neuromodulators based on reward
        rpe = reward - 0.5 # Reward Prediction Error (simple version)
        if rpe > 0:
            self.neuromodulators.update_all(NeuromodulatoryEvent.reward_prediction_error(rpe, 0.5))
        else:
            self.neuromodulators.update_all(NeuromodulatoryEvent.novelty_detection(abs(rpe)))

        # c. Decay neuromodulators
        self.neuromodulators.decay()
        print(f"Neuromodulators updated: {self.neuromodulators.get_all_levels()}")

    def save_state(self, filename: str):
        # ... (Implementation to be added)
        print(f"Saving state to {filename}...")

    def load_state(self, filename: str):
        # ... (Implementation to be added)
        print(f"Loading state from {filename}...")

