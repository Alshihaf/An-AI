"""
SWS Logic - Slow Wave Sleep / System Wide Simulation

This module is responsible for the core decision-making loop, simulating
potential futures and scoring available actions based on internal state
and environmental context. This is the heart of Samre's autonomous drive.
"""
import random
from typing import Dict, List, Any

# A list of all possible high-level actions Samre can take.
POSSIBLE_ACTIONS = [
    "EXPLORE",      # Seek new information, browse files, read content.
    "EVOLVE",       # Improve codebase, synthesize new logic, self-refactor.
    "ORGANIZE",     # Consolidate memories, refactor learned knowledge.
    "REST",         # Do nothing, allow fatigue to decrease and neuromodulators to reset.
    "LEARN",        # Explicitly study a specific file or topic.
    "REASON",       # Perform deductive reasoning on a set of sources.
    # More actions can be added here.
]

def foresight_simulation(
    action: str,
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    ltm_success_rate: float = 0.5, # Placeholder for Long-Term Memory
    cot_confidence: float = 0.5    # Placeholder for Chain of Thought confidence
) -> float:
    """
    Calculates a score for a given action based on the current internal state.
    This simulates the "desirability" of an action in the current context.

    Args:
        action: The action to score (e.g., "EXPLORE").
        needs: A dictionary of current internal need levels.
        neuromodulators: A dictionary of current neuromodulator levels.
        ltm_success_rate: Historical success rate of this action (from LTM).
        cot_confidence: Confidence from a recent CoT if relevant.

    Returns:
        A score from 0.0 to potentially >1.0.
    """
    score = 0.0
    
    # 1. Influence of Internal Needs
    if action == "EXPLORE":
        # Driven by hunger for information and boredom.
        score += needs.get("hunger", 0.0) * 1.5
        score += needs.get("boredom", 0.0) * 1.0
        score -= needs.get("fatigue", 0.0) * 0.5 # Less likely to explore if tired
    elif action == "EVOLVE":
        # Driven by a need to fix "messiness" in its own code.
        score += needs.get("messiness", 0.0) * 1.2
        score -= needs.get("fatigue", 0.0) * 0.5
    elif action == "ORGANIZE":
        # Driven by messiness, but less intense than evolving.
        score += needs.get("messiness", 0.0) * 0.8
        score += needs.get("boredom", 0.0) * 0.5
    elif action == "LEARN":
        # A more focused version of EXPLORE.
        score += needs.get("hunger", 0.0) * 1.2
    elif action == "REASON":
        # A high-cognitive-cost action.
        score += needs.get("hunger", 0.0) * 0.8
        score -= needs.get("fatigue", 0.0) * 0.8 # Very unlikely if tired
    elif action == "REST":
        # Driven strongly by fatigue.
        score += needs.get("fatigue", 0.0) * 2.0
        # Also a fallback if other needs are low.
        if all(n < 0.2 for n in needs.values()):
            score += 0.3

    # 2. Influence of Neuromodulators
    dopamine = neuromodulators.get("dopamine", 0.5) # Motivation
    serotonin = neuromodulators.get("serotonin", 0.5) # Mood/Contentment
    cortisol = neuromodulators.get("cortisol", 0.1)   # Stress

    # Dopamine boosts goal-oriented actions
    if action not in ["REST"]:
        score *= (1.0 + dopamine * 0.5)

    # High serotonin makes resting more attractive
    if action == "REST":
        score *= (1.0 + serotonin * 0.3)
        
    # Cortisol (stress) can be a motivator for urgent needs, but also hinders complex tasks
    if needs.get("hunger", 0.0) > 0.8 or needs.get("messiness", 0.0) > 0.8:
        score *= (1.0 + cortisol * 0.2)
    if action in ["EVOLVE", "REASON"]:
        score *= (1.0 - cortisol * 0.5) # Stress makes complex thought harder

    # 3. Influence of Learning and Confidence
    # Higher historical success makes an action more likely
    score *= (0.5 + ltm_success_rate) 
    # Higher confidence from prior reasoning boosts related actions
    score *= (0.5 + cot_confidence)

    # 4. Add a small amount of randomness for emergent behavior
    score += random.uniform(-0.05, 0.05)
    
    return max(0, score)

def score_all_actions(
    needs: Dict[str, float],
    neuromodulators: Dict[str, float],
    # In the future, this would also take LTM and CoT context
) -> Dict[str, float]:
    """
    Scores all possible actions and returns a dictionary of {action: score}.
    """
    scores = {}
    for action in POSSIBLE_ACTIONS:
        # For now, LTM and CoT are placeholders
        scores[action] = foresight_simulation(action, needs, neuromodulators)
    return scores
