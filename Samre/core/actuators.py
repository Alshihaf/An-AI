"""
Actuators - Action Execution Components for Samre.

Each class here represents the agent's ability to perform a specific action
in its environment, such as exploring for new information, learning it,
or reasoning about its knowledge.
"""

import os
from typing import Optional, Set, Tuple

from tools.file_manager import FileManager


class ExploreActuator:
    """Finds new, unread files for the agent to learn from."""
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.ignore_dirs = {'.git', '__pycache__', '.idea', 'Samre/log'}
        # Expanded perceptions: now looks for text, code, and common binary files
        self.valid_extensions = {
            '.py', '.md',  # Text & Code
            '.db', '.sqlite', '.bin', '.dat', # Data & Binary
            '.so', '.dll', '.exe',          # Libraries & Executables
            '.png', '.jpg', '.jpeg', '.svg', # Images
            '.zip', '.gz', '.tar',           # Archives
        }

    def execute(self, explored_paths: Set[str]) -> Optional[str]:
        """
        Scans the filesystem for a file that has not been explored yet.

        Args:
            explored_paths: A set of file paths that the agent already knows about.

        Returns:
            The path to a new file, or None if no new files are found.
        """
        print("🗺️ EXPLORING: Searching for new information sources...")
        for root, dirs, files in os.walk(self.file_manager.base_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for file in files:
                # Check if the file extension is in our set of valid ones
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.file_manager.base_path)
                    if relative_path not in explored_paths:
                        print(f"    ✅ Found new file: {relative_path}")
                        return relative_path
        
        print("    ⚠️ No new files found to explore.")
        return None


class LearningActuator:
    """Reads a specific file to provide its content for learning."""
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def execute_text(self, target_file: str) -> Optional[Tuple[str, str]]:
        """
        Reads a file assuming it is text-based.
        Returns a tuple of (content, path) on success, or None on failure.
        """
        print(f"📚 LEARNING (Text): Attempting to learn from '{target_file}'.")
        read_result = self.file_manager.read_file(target_file)
        if "content" in read_result:
            print(f"    ✅ Success: Read text content from '{target_file}'.")
            return read_result["content"], target_file
        else:
            print(f"    ❌ Failure: Could not read '{target_file}' as text.")
            return None


class ReasoningActuator:
    """Analyzes the knowledge garden to draw conclusions (conceptual)."""
    def execute(self) -> bool:
        print("🤔 REASONING: Analyzing knowledge...")
        print("    ✅ Conclusion: The agent reflects on its knowledge.")
        return True


class EvolutionaryActuator:
    """Analyzes own source code to prepare for self-modification (conceptual)."""
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def execute(self, target_file: str = "core/flock_of_thought.py") -> bool:
        print(f"🧬 EVOLVING: Preparing evolution by analyzing '{target_file}'.")
        read_result = self.file_manager.read_file(target_file)
        if "content" in read_result:
            print(f"    ✅ Analysis: Successfully read {len(read_result['content'])} characters.")
            return True
        else:
            print(f"    ❌ Failure: Could not read source code '{target_file}'.")
            return False
