"""
Memory Replay Buffer for EMMET
Stores historical edits and provides sampling strategies to mitigate catastrophic forgetting.

Usage:
    buffer = ReplayBuffer(max_size=200, strategy='random')
    buffer.add(request, keys, values)
    sampled = buffer.sample(replay_rate=0.3, current_batch_size=32)
"""

import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

import torch
import numpy as np


logger = logging.getLogger(__name__)


class EditRecord:
    """Single edit record in the replay buffer"""
    
    def __init__(
        self,
        edit_id: int,
        request: Dict,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
        priority: float = 1.0
    ):
        """
        Args:
            edit_id: Unique identifier for this edit
            request: Original edit request dict with subject, prompt, target_new, etc.
            keys: Cached key statistics (optional, for acceleration)
            values: Cached value statistics (optional, for acceleration)
            timestamp: Time when edit was added
            priority: Priority score for sampling (default 1.0)
        """
        self.edit_id = edit_id
        self.request = request
        self.keys = keys
        self.values = values
        self.timestamp = timestamp or time.time()
        self.priority = priority
        self.access_count = 0  # For LRU tracking
        
    def __repr__(self):
        subject = self.request.get('subject', 'N/A')
        target = self.request.get('target_new', {}).get('str', 'N/A')
        return f"EditRecord(id={self.edit_id}, subject='{subject}', target='{target}')"


class ReplayBuffer:
    """
    Memory Replay Buffer with multiple sampling strategies
    
    Supports:
    - Random sampling: uniform probability
    - Priority sampling: weighted by priority scores
    - Recent sampling: prefer recent edits
    - FIFO/LRU eviction when buffer is full
    """
    
    def __init__(
        self,
        max_size: int = 200,
        strategy: str = 'random',
        eviction: str = 'fifo',
        deduplicate: bool = True
    ):
        """
        Args:
            max_size: Maximum number of edits to store
            strategy: Sampling strategy ('random', 'priority', 'recent')
            eviction: Eviction strategy when full ('fifo', 'lru')
            deduplicate: Whether to remove duplicate edits (same subject)
        """
        self.max_size = max_size
        self.strategy = strategy
        self.eviction = eviction
        self.deduplicate = deduplicate
        
        self.buffer = deque(maxlen=max_size if eviction == 'fifo' else None)
        self.edit_count = 0  # Total edits added (including evicted)
        self.subject_index = {}  # subject -> list of edit_ids for deduplication
        
        logger.info(f"ReplayBuffer initialized: max_size={max_size}, "
                   f"strategy={strategy}, eviction={eviction}, deduplicate={deduplicate}")
    
    def add(
        self,
        request: Dict,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        priority: float = 1.0
    ) -> EditRecord:
        """
        Add a new edit to the buffer
        
        Args:
            request: Edit request dictionary
            keys: Cached key statistics (optional)
            values: Cached value statistics (optional)
            priority: Priority score for this edit
            
        Returns:
            The created EditRecord
        """
        # Create record
        record = EditRecord(
            edit_id=self.edit_count,
            request=request,
            keys=keys.detach().cpu() if keys is not None else None,
            values=values.detach().cpu() if values is not None else None,
            priority=priority
        )
        
        # Handle deduplication
        subject = request.get('subject', '')
        if self.deduplicate and subject:
            if subject in self.subject_index:
                # Remove old edit with same subject
                old_ids = self.subject_index[subject]
                self._remove_by_ids(old_ids)
                logger.debug(f"Deduplicated: removed {len(old_ids)} old edits for subject '{subject}'")
            self.subject_index[subject] = [record.edit_id]
        
        # Add to buffer
        if self.eviction == 'fifo':
            # deque with maxlen handles eviction automatically
            if len(self.buffer) >= self.max_size:
                evicted = self.buffer[0]
                logger.debug(f"FIFO eviction: {evicted}")
            self.buffer.append(record)
        elif self.eviction == 'lru':
            # Manual LRU eviction
            if len(self.buffer) >= self.max_size:
                # Find least recently accessed
                lru_record = min(self.buffer, key=lambda r: r.access_count)
                self.buffer.remove(lru_record)
                logger.debug(f"LRU eviction: {lru_record}")
            self.buffer.append(record)
        
        self.edit_count += 1
        
        if self.edit_count % 100 == 0:
            logger.info(f"ReplayBuffer: added {self.edit_count} edits, current size={len(self.buffer)}")
        
        return record
    
    def sample(
        self,
        replay_rate: float,
        current_batch_size: int,
        exclude_recent: int = 0
    ) -> List[EditRecord]:
        """
        Sample edits from the buffer
        
        Args:
            replay_rate: Proportion of replay samples (0.0-1.0)
            current_batch_size: Size of current editing batch
            exclude_recent: Exclude the N most recent edits from sampling
            
        Returns:
            List of sampled EditRecords
        """
        if len(self.buffer) == 0:
            logger.warning("ReplayBuffer is empty, returning empty sample")
            return []
        
        # Calculate number of samples
        num_samples = int(replay_rate * current_batch_size)
        num_samples = min(num_samples, len(self.buffer) - exclude_recent)
        
        if num_samples <= 0:
            return []
        
        # Get candidate pool (excluding recent if specified)
        if exclude_recent > 0:
            candidates = list(self.buffer)[:-exclude_recent]
        else:
            candidates = list(self.buffer)
        
        if len(candidates) == 0:
            return []
        
        # Sample based on strategy
        if self.strategy == 'random':
            sampled = random.sample(candidates, min(num_samples, len(candidates)))
        
        elif self.strategy == 'priority':
            # Weighted sampling by priority
            weights = np.array([r.priority for r in candidates])
            weights = weights / weights.sum()  # Normalize
            indices = np.random.choice(
                len(candidates),
                size=min(num_samples, len(candidates)),
                replace=False,
                p=weights
            )
            sampled = [candidates[i] for i in indices]
        
        elif self.strategy == 'recent':
            # Sample more recent edits with higher probability
            # Use exponential decay: weight = exp(-α * age_rank)
            alpha = 0.1
            ages = [(time.time() - r.timestamp) for r in candidates]
            weights = np.exp(-alpha * np.argsort(np.argsort(ages)))
            weights = weights / weights.sum()
            indices = np.random.choice(
                len(candidates),
                size=min(num_samples, len(candidates)),
                replace=False,
                p=weights
            )
            sampled = [candidates[i] for i in indices]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
        
        # Update access counts for LRU
        for record in sampled:
            record.access_count += 1
        
        logger.debug(f"Sampled {len(sampled)} edits (rate={replay_rate}, batch={current_batch_size})")
        
        return sampled
    
    def get_requests(self, records: List[EditRecord]) -> List[Dict]:
        """Extract request dicts from records"""
        return [r.request for r in records]
    
    def get_cached_stats(self, records: List[EditRecord]) -> Tuple[Optional[List], Optional[List]]:
        """
        Extract cached keys and values from records
        
        Returns:
            Tuple of (keys_list, values_list) or (None, None) if not cached
        """
        if not records:
            return None, None
        
        # Check if all records have cached stats
        has_keys = all(r.keys is not None for r in records)
        has_values = all(r.values is not None for r in records)
        
        if has_keys and has_values:
            keys = [r.keys for r in records]
            values = [r.values for r in records]
            return keys, values
        else:
            return None, None
    
    def _remove_by_ids(self, edit_ids: List[int]):
        """Remove records by their edit IDs"""
        to_remove = [r for r in self.buffer if r.edit_id in edit_ids]
        for record in to_remove:
            self.buffer.remove(record)
    
    def size(self) -> int:
        """Current buffer size"""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def clear(self):
        """Clear all records"""
        self.buffer.clear()
        self.subject_index.clear()
        logger.info("ReplayBuffer cleared")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'total_edits_added': self.edit_count,
            'strategy': self.strategy,
            'eviction': self.eviction,
            'deduplicate': self.deduplicate,
            'unique_subjects': len(self.subject_index)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        return f"ReplayBuffer(size={len(self)}/{self.max_size}, strategy={self.strategy})"
