from typing import Any, List
import torch


def get_default_device() -> torch.device:
    """Select the best available PyTorch device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def calculate_expected_value(probs: torch.Tensor) -> float:
    """
    Calculate the expected value of a probability distribution.
    """
    tokens = torch.arange(len(probs), device=probs.device)
    ev = torch.sum(tokens * probs)
    return ev.item()

def calculate_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy in bits; clamps probabilities to avoid log(0)."""
    safe_probs = torch.clamp(probs, min=1e-12, max=1.0)
    entropy = -torch.sum(safe_probs * torch.log2(safe_probs))
    return float(entropy.item())

def calculate_normalised_entropy(probs: torch.Tensor) -> float:
    """Entropy divided by log₂ N so the result is in [0, 1]."""
    n = len(probs)
    max_entropy_tensor = torch.log2(torch.tensor(float(n), device=probs.device))
    max_entropy = float(max_entropy_tensor.item())
    return calculate_entropy(probs) / max_entropy if max_entropy > 0 else 0.0


def calculate_effective_support_size(probs: torch.Tensor) -> float:
    """Effective number of outcomes carrying probability mass (2 ** entropy)."""
    return float(2 ** calculate_entropy(probs))


def calculate_topk_mass(probs: torch.Tensor, k: int = 3) -> float:
    """Total probability contained in the top-k most likely bins."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    sorted_probs = torch.sort(probs, descending=True)[0]
    topk_sum = torch.sum(sorted_probs[:k])
    return float(topk_sum.item())


def calculate_kl_divergence_uniform(probs: torch.Tensor) -> float:
    """KL divergence D_KL(p || uniform)."""
    n = len(probs)
    safe_probs = torch.clamp(probs, min=1e-12, max=1.0)
    uniform_prob = torch.tensor(1.0 / n, device=probs.device)
    kl = torch.sum(safe_probs * (torch.log2(safe_probs) - torch.log2(uniform_prob)))
    return float(kl.item())

def calculate_mutual_information(distributions: List[torch.Tensor]) -> float:
    """Mutual information I(X; M) in bits between outcome and model index."""
    if not distributions:
        raise ValueError("Distributions list cannot be empty")

    stacked = torch.stack(distributions)
    mean_dist = torch.mean(stacked, dim=0)
    h_mix = calculate_entropy(mean_dist)
    individual_entropies = [calculate_entropy(d) for d in distributions]
    h_individual = sum(individual_entropies) / len(distributions)
    mi = h_mix - h_individual
    return float(mi)

def infer_confidence(probs: torch.Tensor) -> str:
    """Map normalised entropy to qualitative confidence buckets."""
    norm_ent = calculate_normalised_entropy(probs)
    if norm_ent < 0.2:
        return "HIGH confidence"
    if norm_ent < 0.4:
        return "MEDIUM-HIGH confidence"
    if norm_ent < 0.6:
        return "MEDIUM confidence"
    if norm_ent < 0.8:
        return "LOW-MEDIUM confidence"
    return "LOW confidence"

def infer_likelihood(data: torch.Tensor) -> str:
    """Translate expected value into qualitative likelihood labels."""
    expected_value = calculate_expected_value(data)
    if expected_value > 95:
        return "Almost certain"
    if expected_value > 80:
        return "Highly likely"
    if expected_value > 55:
        return "Likely or probably"
    if expected_value > 40:
        return "Realistic possibility"
    if expected_value > 25:
        return "Unlikely"
    if expected_value > 10:
        return "Highly unlikely"
    return "Remote chance"
