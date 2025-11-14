"""
EMMET with Memory Replay Integration

This module extends the original EMMET implementation with Memory Replay capability.
It wraps the apply_emmet_to_model function to automatically manage replay buffer.

Usage:
    from emmet.emmet_replay import apply_emmet_with_replay
    
    model, weights_copy, distances = apply_emmet_with_replay(
        model, tok, requests, hparams,
        use_replay=True,
        replay_rate=0.3,
        replay_buffer_size=200
    )
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .emmet_main import apply_emmet_to_model, execute_emmet
from .emmet_hparams import EMMETHyperParams
from .replay_buffer import ReplayBuffer, EditRecord
from .replay_utils import (
    merge_requests,
    check_dimension_compatibility,
    apply_tikhonov_regularization,
    check_condition_number,
    log_replay_stats,
    validate_replay_integration,
    compute_edit_priority
)


logger = logging.getLogger(__name__)


# Global replay buffer instance
_REPLAY_BUFFER = None


def get_replay_buffer(
    max_size: int = 200,
    strategy: str = 'random',
    eviction: str = 'fifo',
    deduplicate: bool = True,
    reset: bool = False
) -> ReplayBuffer:
    """
    Get or create the global replay buffer
    
    Args:
        max_size: Maximum buffer size
        strategy: Sampling strategy
        eviction: Eviction strategy
        deduplicate: Enable deduplication
        reset: Force reset the buffer
    
    Returns:
        ReplayBuffer instance
    """
    global _REPLAY_BUFFER
    
    if _REPLAY_BUFFER is None or reset:
        _REPLAY_BUFFER = ReplayBuffer(
            max_size=max_size,
            strategy=strategy,
            eviction=eviction,
            deduplicate=deduplicate
        )
        logger.info(f"Created new replay buffer: {_REPLAY_BUFFER}")
    
    return _REPLAY_BUFFER


def apply_emmet_with_replay(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EMMETHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    # Replay-specific parameters
    use_replay: bool = False,
    replay_rate: float = 0.3,
    replay_buffer_size: int = 200,
    replay_strategy: str = 'random',
    replay_weight: float = 1.0,
    store_cache: bool = False,
    enable_tikhonov: bool = True,
    lambda_reg: float = 1e-5,
    check_stability: bool = True
) -> Tuple[AutoModelForCausalLM, Dict[str, Any], Dict[str, Any]]:
    """
    Apply EMMET with optional Memory Replay
    
    Args:
        model: Model to edit
        tok: Tokenizer
        requests: Current batch of edit requests
        hparams: EMMET hyperparameters
        copy: Whether to copy the model
        return_orig_weights: Whether to return original weights
        cache_template: Cache template path
        
        # Replay parameters
        use_replay: Enable replay mechanism
        replay_rate: Proportion of replay samples (0.0-1.0)
        replay_buffer_size: Maximum buffer size
        replay_strategy: Sampling strategy ('random', 'priority', 'recent')
        replay_weight: Weight for replay samples (0.0-1.0)
        store_cache: Store keys/values in buffer for acceleration
        enable_tikhonov: Apply Tikhonov regularization
        lambda_reg: Regularization strength
        check_stability: Check numerical stability
    
    Returns:
        Tuple of (model, weights_copy, distances)
    """
    
    if not use_replay:
        # Standard EMMET without replay
        logger.info("Running standard EMMET (no replay)")
        return apply_emmet_to_model(
            model, tok, requests, hparams,
            copy=copy,
            return_orig_weights=return_orig_weights,
            cache_template=cache_template
        )
    
    # Get or create replay buffer
    buffer = get_replay_buffer(
        max_size=replay_buffer_size,
        strategy=replay_strategy,
        eviction='fifo',
        deduplicate=True
    )
    
    logger.info(f"Running EMMET with Memory Replay: buffer_size={buffer.size()}, "
               f"replay_rate={replay_rate}, strategy={replay_strategy}")
    
    # Sample from replay buffer
    replay_records = buffer.sample(
        replay_rate=replay_rate,
        current_batch_size=len(requests),
        exclude_recent=0  # Can exclude very recent edits if needed
    )
    
    replay_requests = buffer.get_requests(replay_records)
    
    if len(replay_requests) > 0:
        # Merge current batch with replay samples
        merged_requests, sample_weights = merge_requests(
            requests,
            replay_requests,
            replay_weight=replay_weight
        )
        
        log_replay_stats(
            current_size=len(requests),
            replay_size=len(replay_requests),
            buffer_size=buffer.size(),
            replay_rate=replay_rate
        )
        
        # Validate merge
        if not validate_replay_integration(
            len(requests),
            len(merged_requests),
            len(replay_requests)
        ):
            logger.error("Replay integration validation failed, falling back to no replay")
            merged_requests = requests
    else:
        logger.info("Buffer empty or no replay samples, using current batch only")
        merged_requests = requests
    
    # Apply EMMET with merged batch
    try:
        model, weights_copy, distances = apply_emmet_to_model(
            model, tok, merged_requests, hparams,
            copy=copy,
            return_orig_weights=return_orig_weights,
            cache_template=cache_template
        )
        
        # Add current requests to replay buffer for future use
        for request in requests:
            # Compute priority (could be based on edit success, but we use 1.0 for now)
            priority = compute_edit_priority(request, success_score=1.0)
            
            # Add to buffer (with optional cached keys/values)
            buffer.add(
                request=request,
                keys=None,  # TODO: optionally cache computed keys
                values=None,  # TODO: optionally cache computed values
                priority=priority
            )
        
        logger.info(f"Added {len(requests)} new edits to buffer (new size: {buffer.size()})")
        
        # Log buffer stats
        stats = buffer.get_stats()
        logger.info(f"Buffer stats: {stats}")
        
        return model, weights_copy, distances
    
    except Exception as e:
        logger.error(f"EMMET with replay failed: {e}", exc_info=True)
        raise


def reset_replay_buffer():
    """Reset the global replay buffer"""
    global _REPLAY_BUFFER
    if _REPLAY_BUFFER is not None:
        _REPLAY_BUFFER.clear()
        logger.info("Replay buffer reset")


def get_buffer_stats() -> Optional[Dict]:
    """Get statistics from the global replay buffer"""
    global _REPLAY_BUFFER
    if _REPLAY_BUFFER is not None:
        return _REPLAY_BUFFER.get_stats()
    return None
