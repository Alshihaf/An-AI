#tools/binary_reader.py
import sys
import re
import math
from typing import List, Optional

# ----------------------------------------------------------------------
# Pure vector operations (optimised where possible)
# ----------------------------------------------------------------------

def binary_str_to_vector(bin_str: str) -> List[int]:
    """Convert binary string to list of integers (0/1)."""
    return [1 if ch == '1' else 0 for ch in bin_str]

def vector_length(v: List[int]) -> int:
    return len(v)

def count_ones(v: List[int]) -> int:
    return sum(v)

def proportion_ones(v: List[int]) -> float:
    n = len(v)
    return sum(v) / n if n else 0.0

def run_lengths(v: List[int]) -> List[int]:
    if not v:
        return []
    runs = []
    current = v[0]
    length = 1
    for bit in v[1:]:
        if bit == current:
            length += 1
        else:
            runs.append(length)
            current = bit
            length = 1
    runs.append(length)
    return runs

def autocorrelation(v: List[int], max_lag: Optional[int] = None) -> List[float]:
    """
    Optimised autocorrelation using incremental sums.
    Complexity O(n * max_lag) but with lower constant factors.
    For very long vectors, max_lag is capped at 64.
    """
    n = len(v)
    if n == 0:
        return []
    if max_lag is None:
        max_lag = min(n - 1, 64)   # cap for performance
    else:
        max_lag = min(max_lag, n - 1)

    mean = sum(v) / n
    # Pre‑compute centred values
    centred = [x - mean for x in v]

    # Variance (unbiased estimate, but we use population variance for lag adjustment)
    var = sum(c * c for c in centred)   # population variance = sum((x-mean)^2)
    if var == 0:
        return [1.0] + [0.0] * max_lag

    result = [1.0]   # lag 0
    for lag in range(1, max_lag + 1):
        # cov = sum(centred[i] * centred[i - lag] for i in range(lag, n))
        # Optimisation: use Python's sum with generator
        cov = sum(centred[i] * centred[i - lag] for i in range(lag, n))
        # Normalize by (n - lag) * variance
        corr = cov / ((n - lag) * var)
        result.append(corr)
    return result

def shannon_entropy(v: List[int]) -> float:
    n = len(v)
    if n == 0:
        return 0.0
    p1 = sum(v) / n
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 0:
        ent -= p0 * math.log2(p0)
    if p1 > 0:
        ent -= p1 * math.log2(p1)
    return ent

def mean_run_length(v: List[int]) -> float:
    runs = run_lengths(v)
    if not runs:
        return 0.0
    return sum(runs) / len(runs)

# ----------------------------------------------------------------------
# Vector‑based silent categorization (knowledge without language)
# ----------------------------------------------------------------------

# Internal category constants (pure integers)
CAT_LOW_ENTROPY_REPETITIVE = 0   # e.g., alternating patterns, long runs
CAT_HIGH_ENTROPY_RANDOM     = 1   # high entropy, short runs, low autocorrelation
CAT_SILENCE_OR_CONSTANT     = 2   # all zeros or all ones
CAT_MEDIUM_ENTROPY_STRUCT   = 3   # moderate entropy, possible structured data
CAT_BIASED_HIGH_ENTROPY     = 4   # NEW: distribution far from 0.5 yet high entropy
CAT_UNKNOWN                 = 5

def infer_vector_category(v: List[int]) -> int:
    """
    Infer the nature of the binary vector using raw vector metrics.
    Returns an integer category code.
    """
    n = len(v)
    if n == 0:
        return CAT_UNKNOWN

    ones_prop = proportion_ones(v)

    # Constant vector detection
    if ones_prop == 0.0 or ones_prop == 1.0:
        return CAT_SILENCE_OR_CONSTANT

    # For very short vectors, fallback to simple run length
    if n < 8:
        mean_run = mean_run_length(v)
        if mean_run <= 1.5:
            return CAT_HIGH_ENTROPY_RANDOM
        else:
            return CAT_LOW_ENTROPY_REPETITIVE

    entropy = shannon_entropy(v)
    mean_run = mean_run_length(v)

    # Distribution deviation from perfect balance
    deviation = abs(ones_prop - 0.5)

    # Autocorrelation at lag 1 (measure of short‑term memory)
    autocorr = autocorrelation(v, max_lag=1)
    lag1_corr = autocorr[1] if len(autocorr) > 1 else 0.0

    # --- Enhanced decision tree ---

    # 1. High entropy but strongly biased distribution
    if entropy > 0.9 and deviation > 0.2:
        return CAT_BIASED_HIGH_ENTROPY

    # 2. Genuinely random-like: high entropy, near 0.5 balance, short runs, low lag‑1 correlation
    if entropy > 0.95 and deviation < 0.05 and mean_run < 2.5 and abs(lag1_corr) < 0.1:
        return CAT_HIGH_ENTROPY_RANDOM

    # 3. Highly repetitive / low entropy
    if entropy < 0.3:
        return CAT_LOW_ENTROPY_REPETITIVE

    # 4. Structured medium entropy: moderate entropy, runs not too short, some correlation
    if 0.4 <= entropy <= 0.85 and 2.0 <= mean_run <= 6.0:
        return CAT_MEDIUM_ENTROPY_STRUCT

    # 5. Edge cases that still show repetitive character due to long runs
    if mean_run > 5.0 and lag1_corr > 0.4:
        return CAT_LOW_ENTROPY_REPETITIVE

    # 6. Otherwise, if we cannot confidently classify, it remains UNKNOWN
    return CAT_UNKNOWN

# ----------------------------------------------------------------------
# Internal knowledge accumulator (state that "knows")
# ----------------------------------------------------------------------

class SilentKnowledge:
    """
    Internal state that accumulates what the program "knows" about the data.
    All attributes are plain numbers; no human‑readable strings.
    """
    def __init__(self):
        self.category_counts = [0] * 6       # now 6 categories (0..5)
        self.total_vectors = 0
        self.total_bits = 0
        self.cumulative_entropy = 0.0
        self.sum_proportion_ones = 0.0
        self.sum_mean_run = 0.0
        self.sum_deviation = 0.0

    def update(self, v: List[int], category: int):
        self.total_vectors += 1
        self.category_counts[category] += 1
        n = len(v)
        self.total_bits += n
        self.cumulative_entropy += shannon_entropy(v)
        p1 = proportion_ones(v)
        self.sum_proportion_ones += p1
        self.sum_mean_run += mean_run_length(v)
        self.sum_deviation += abs(p1 - 0.5)

# Global knowledge instance (only accessible internally)
_knowledge = SilentKnowledge()

# ----------------------------------------------------------------------
# Core analysis that "thinks" and now also "knows"
# ----------------------------------------------------------------------

def analyze_raw_vector(v: List[int]) -> None:
    """
    Perform pure vector computations and internally categorize the vector.
    The knowledge is stored in the global _knowledge object.
    No output is produced.
    """
    if not v:
        return

    # --- basic aggregates (computed for their own sake) ---
    _ = vector_length(v)
    _ = count_ones(v)
    _ = proportion_ones(v)
    _ = run_lengths(v)
    _ = autocorrelation(v, max_lag=min(len(v)-1, 64))
    _ = shannon_entropy(v)

    # --- inference of category (the "knowing" part) ---
    cat = infer_vector_category(v)
    _knowledge.update(v, cat)

    # Additional silent thought: XOR with shifted self (just an internal operation)
    if len(v) > 1:
        shifted = v[1:] + [v[0]]
        _ = [a ^ b for a, b in zip(v, shifted)]

# ----------------------------------------------------------------------
# Extraction of binary strings from raw content
# ----------------------------------------------------------------------

def extract_binary_strings(content: bytes) -> List[str]:
    try:
        text = content.decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        text = content.decode('latin1')
    pattern = re.compile(r'[01]+')
    return pattern.findall(text)

# ----------------------------------------------------------------------
# Main silent entry point
# ----------------------------------------------------------------------

def main() -> None:
    # Read input
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'rb') as f:
                content = f.read()
        except Exception:
            return
    else:
        content = sys.stdin.buffer.read()

    # Extract binary strings
    bin_strings = extract_binary_strings(content)

    # Process each binary string as a raw vector and silently "learn"
    for bs in bin_strings:
        vec = binary_str_to_vector(bs)
        analyze_raw_vector(vec)

    # At this point, _knowledge contains what the program "knows".
    # No output is ever produced.