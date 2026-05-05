"""
Samre - Autonomous AI Agent

This is the main entry point for running the autonomous agent loop.
"""

import sys
import os
import time
import argparse
from typing import Optional

# Add the project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.flock_of_thought import FlockOfThought

class AutonomousAgent:
    """The main class that runs Samre's autonomous lifecycle."""

    def __init__(self):
        self.flock = FlockOfThought()
        self.running = True

    def run_cycle(self):
        """Executes a single cognitive cycle."""
        # Steps 1 & 2: Update needs and score actions
        action_scores = self.flock.update_and_score_actions()
        
        # Step 3: Evaluate and select an action
        selected_action = self.flock.evaluate_and_select_action(action_scores)
        
        # Steps 4, 5, 6: Execute, record, and learn
        self.flock.execute_action(selected_action)

    def start(self, cycle_delay: float = 5.0):
        """Starts the main autonomous loop."""
        print("🚀 Starting Samre's autonomous loop. Press Ctrl+C to stop.")
        try:
            while self.running:
                self.run_cycle()
                print(f"--- CYCLE END ---\nWaiting for {cycle_delay} seconds...")
                time.sleep(cycle_delay)
        except KeyboardInterrupt:
            print("\n🛑 Loop interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Handles the shutdown process."""
        print("👋 Shutting down...")
        self.running = False
        print("✅ Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Run the Samre Autonomous Agent.")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between cycles.")
    args = parser.parse_args()

    agent = AutonomousAgent()
    agent.start(cycle_delay=args.delay)

if __name__ == "__main__":
    main()
