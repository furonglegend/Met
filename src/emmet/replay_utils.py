"""
Utility functions for Memory Replay integration with EMMET

Provides helper functions for:
- Merging current batch with replay samples
- Numerical stability checks
- Dimension alignment
- Weight adjustment for replay samples
"""

import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np


logger = logging.getLogger(__name__)


def merge_requests(
    current_requests: List[Dict],
    replay_requests: List[Dict],
    replay_weight: float = 1.0
) -> Tuple[List[Dict], List[float]]:
    """
    Merge current batch requests with replay samples
    
    Args:
        current_requests: Current editing batch
        replay_requests: Sampled replay edits
        replay_weight: Weight multiplier for replay samples (0.0-1.0)
                      Set <1.0 to down-weight historical edits
    
    Returns:
        Tuple of (merged_requests, sample_weights)
    """
    merged = current_requests + replay_requests
    
    # Assign weights: current edits get weight 1.0, replay edits get replay_weight
    weights = [1.0] * len(current_requests) + [replay_weight] * len(replay_requests)
    
    logger.debug(f"Merged {len(current_requests)} current + {len(replay_requests)} replay "
                f"= {len(merged)} total (replay_weight={replay_weight})")
    
    return merged, weights


def check_dimension_compatibility(
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    replay_keys: Optional[List[torch.Tensor]],
    replay_values: Optional[List[torch.Tensor]]
) -> bool:
    """
    Check if replay samples have compatible dimensions with current batch
    
    Args:
        current_keys: Keys from current batch [batch, hidden_dim]
        current_values: Values from current batch [batch, hidden_dim]
        replay_keys: List of cached keys from replay samples
        replay_values: List of cached values from replay samples
    
    Returns:
        True if compatible, False otherwise
    """
    if replay_keys is None or replay_values is None:
        return True  # No cached stats to check
    
    try:
        for rk, rv in zip(replay_keys, replay_values):
            if rk.shape[-1] != current_keys.shape[-1]:
                logger.warning(f"Key dimension mismatch: replay {rk.shape} vs current {current_keys.shape}")
                return False
            if rv.shape[-1] != current_values.shape[-1]:
                logger.warning(f"Value dimension mismatch: replay {rv.shape} vs current {current_values.shape}")
                return False
        return True
    except Exception as e:
        logger.error(f"Dimension check failed: {e}")
        return False


def apply_tikhonov_regularization(
    matrix: torch.Tensor,
    lambda_reg: float = 1e-5
) -> torch.Tensor:
    """
    Apply Tikhonov regularization to improve numerical stability
    
    Adds λI to the matrix before inversion to prevent singularity
    
    Args:
        matrix: Square matrix to regularize
        lambda_reg: Regularization strength
    
    Returns:
        Regularized matrix
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    regularized = matrix + lambda_reg * identity
    
    logger.debug(f"Applied Tikhonov regularization with λ={lambda_reg}")
    
    return regularized


def check_condition_number(
    matrix: torch.Tensor,
    threshold: float = 1e10,
    name: str = "matrix"
) -> Tuple[bool, float]:
    """
    Check the condition number of a matrix
    
    High condition numbers indicate numerical instability
    
    Args:
        matrix: Matrix to check
        threshold: Condition number threshold for warning
        name: Name for logging
    
    Returns:
        Tuple of (is_stable, condition_number)
    """
    try:
        cond_num = torch.linalg.cond(matrix).item()
        is_stable = cond_num < threshold
        
        if not is_stable:
            logger.warning(f"{name} has high condition number: {cond_num:.2e} (threshold={threshold:.2e})")
        else:
            logger.debug(f"{name} condition number: {cond_num:.2e} (stable)")
        
        return is_stable, cond_num
    
    except Exception as e:
        logger.error(f"Failed to compute condition number for {name}: {e}")
        return False, float('inf')


def adaptive_merge_strategy(
    current_batch_size: int,
    buffer_size: int,
    replay_rate: float,
    min_replay: int = 5,
    max_replay: int = 100
) -> int:
    """
    Adaptively determine number of replay samples based on buffer state
    
    Args:
        current_batch_size: Size of current batch
        buffer_size: Current replay buffer size
        replay_rate: Base replay rate (0.0-1.0)
        min_replay: Minimum replay samples
        max_replay: Maximum replay samples
    
    Returns:
        Number of replay samples to use
    """
    # Calculate based on replay_rate
    num_replay = int(replay_rate * current_batch_size)
    
    # Constrain by buffer size
    num_replay = min(num_replay, buffer_size)
    
    # Apply min/max bounds
    num_replay = max(min_replay, min(num_replay, max_replay))
    
    # Don't exceed buffer size
    num_replay = min(num_replay, buffer_size)
    
    logger.debug(f"Adaptive merge: batch={current_batch_size}, buffer={buffer_size}, "
                f"rate={replay_rate} → {num_replay} replay samples")
    
    return num_replay


def compute_edit_priority(
    request: Dict,
    success_score: float = 1.0,
    importance_multiplier: float = 1.0
) -> float:
    """
    Compute priority score for an edit (for priority-based sampling)
    
    Args:
        request: Edit request
        success_score: Success metric (e.g., ES score) from evaluation
        importance_multiplier: Manual importance weight
    
    Returns:
        Priority score (higher = more important)
    """
    # Base priority on success
    priority = success_score
    
    # Apply importance multiplier
    priority *= importance_multiplier
    
    # Could add more factors:
    # - Subject frequency (rarer subjects get higher priority)
    # - Edit difficulty
    # - Temporal decay
    
    return max(0.1, priority)  # Ensure minimum priority


def safe_concatenate(
    tensors: List[torch.Tensor],
    dim: int = 0,
    name: str = "tensors"
) -> Optional[torch.Tensor]:
    """
    Safely concatenate tensors with error handling
    
    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate
        name: Name for logging
    
    Returns:
        Concatenated tensor or None if failed
    """
    if not tensors:
        logger.warning(f"No {name} to concatenate")
        return None
    
    try:
        # Check all tensors have same device and dtype
        device = tensors[0].device
        dtype = tensors[0].dtype
        
        for i, t in enumerate(tensors):
            if t.device != device:
                logger.warning(f"{name}[{i}] on different device: {t.device} vs {device}")
                tensors[i] = t.to(device)
            if t.dtype != dtype:
                logger.warning(f"{name}[{i}] has different dtype: {t.dtype} vs {dtype}")
                tensors[i] = t.to(dtype)
        
        result = torch.cat(tensors, dim=dim)
        logger.debug(f"Concatenated {len(tensors)} {name} → shape {result.shape}")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to concatenate {name}: {e}")
        return None


def log_replay_stats(
    current_size: int,
    replay_size: int,
    buffer_size: int,
    replay_rate: float
):
    """
    Log statistics about replay merging
    
    Args:
        current_size: Number of current edits
        replay_size: Number of replay edits
        buffer_size: Total buffer size
        replay_rate: Replay rate used
    """
    total = current_size + replay_size
    actual_rate = replay_size / total if total > 0 else 0.0
    
    logger.info(f"Replay Stats: current={current_size}, replay={replay_size}, "
               f"buffer={buffer_size}, requested_rate={replay_rate:.2f}, "
               f"actual_rate={actual_rate:.2f}")


def validate_replay_integration(
    original_batch_size: int,
    merged_batch_size: int,
    expected_replay_size: int
) -> bool:
    """
    Validate that replay integration was successful
    
    Args:
        original_batch_size: Original batch size before replay
        merged_batch_size: Merged batch size after replay
        expected_replay_size: Expected number of replay samples
    
    Returns:
        True if validation passed
    """
    expected_total = original_batch_size + expected_replay_size
    
    if merged_batch_size != expected_total:
        logger.error(f"Replay integration validation failed: "
                    f"merged_size={merged_batch_size} != "
                    f"expected={expected_total} (original={original_batch_size} + "
                    f"replay={expected_replay_size})")
        return False
    
    logger.debug(f"Replay integration validated: {merged_batch_size} total "
                f"({original_batch_size} current + {expected_replay_size} replay)")
    
    return True
