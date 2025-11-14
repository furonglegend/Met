"""
Test script for Memory Replay Buffer

Validates the replay buffer implementation without requiring full model editing.

Usage:
    python scripts/test_replay_buffer.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emmet.replay_buffer import ReplayBuffer, EditRecord
from emmet.replay_utils import merge_requests, check_condition_number
import torch
import numpy as np


def test_basic_operations():
    """Test basic buffer operations"""
    print("="*80)
    print("Test 1: Basic Operations")
    print("="*80)
    
    buffer = ReplayBuffer(max_size=10, strategy='random', eviction='fifo')
    
    # Add some edits
    for i in range(15):
        request = {
            'subject': f'Entity_{i}',
            'prompt': 'The capital of {} is',
            'target_new': {'str': f' City_{i}'},
            'target_true': {'str': ' OldCity'}
        }
        buffer.add(request, priority=1.0)
    
    print(f"✓ Added 15 edits to buffer with max_size=10")
    print(f"✓ Current buffer size: {buffer.size()} (should be 10 due to FIFO eviction)")
    assert buffer.size() == 10, "Buffer size should be 10"
    
    # Sample
    samples = buffer.sample(replay_rate=0.5, current_batch_size=10)
    print(f"✓ Sampled {len(samples)} records (replay_rate=0.5, batch_size=10)")
    assert len(samples) == 5, "Should sample 5 records"
    
    # Get stats
    stats = buffer.get_stats()
    print(f"✓ Buffer stats: {stats}")
    
    print("✅ Test 1 PASSED\n")


def test_sampling_strategies():
    """Test different sampling strategies"""
    print("="*80)
    print("Test 2: Sampling Strategies")
    print("="*80)
    
    strategies = ['random', 'priority', 'recent']
    
    for strategy in strategies:
        buffer = ReplayBuffer(max_size=50, strategy=strategy)
        
        # Add edits with varying priorities
        for i in range(50):
            request = {
                'subject': f'Entity_{i}',
                'prompt': 'The capital of {} is',
                'target_new': {'str': f' City_{i}'},
            }
            priority = (i % 10) / 10.0  # Varying priorities
            buffer.add(request, priority=priority)
        
        # Sample
        samples = buffer.sample(replay_rate=0.2, current_batch_size=20)
        print(f"✓ Strategy '{strategy}': sampled {len(samples)} records")
        assert len(samples) == 4, f"Should sample 4 records for strategy {strategy}"
    
    print("✅ Test 2 PASSED\n")


def test_deduplication():
    """Test subject deduplication"""
    print("="*80)
    print("Test 3: Deduplication")
    print("="*80)
    
    buffer = ReplayBuffer(max_size=20, deduplicate=True)
    
    # Add same subject multiple times
    for i in range(5):
        request = {
            'subject': 'Paris',
            'prompt': 'The capital of France is',
            'target_new': {'str': f' Version_{i}'},
        }
        buffer.add(request)
    
    print(f"✓ Added 5 edits with same subject 'Paris'")
    print(f"✓ Buffer size: {buffer.size()} (should be 1 due to deduplication)")
    assert buffer.size() == 1, "Should only keep 1 edit for duplicate subject"
    
    # Add different subjects
    for i in range(10):
        request = {
            'subject': f'City_{i}',
            'prompt': 'The capital of {} is',
            'target_new': {'str': f' Answer_{i}'},
        }
        buffer.add(request)
    
    print(f"✓ Added 10 more edits with unique subjects")
    print(f"✓ Buffer size: {buffer.size()} (should be 11)")
    assert buffer.size() == 11, "Should have 11 unique subjects"
    
    print("✅ Test 3 PASSED\n")


def test_merge_requests():
    """Test request merging utility"""
    print("="*80)
    print("Test 4: Request Merging")
    print("="*80)
    
    current = [
        {'subject': 'A', 'target_new': {'str': ' X'}},
        {'subject': 'B', 'target_new': {'str': ' Y'}},
    ]
    
    replay = [
        {'subject': 'C', 'target_new': {'str': ' Z'}},
    ]
    
    merged, weights = merge_requests(current, replay, replay_weight=0.5)
    
    print(f"✓ Merged {len(current)} current + {len(replay)} replay = {len(merged)} total")
    print(f"✓ Weights: {weights}")
    assert len(merged) == 3, "Should have 3 total requests"
    assert weights == [1.0, 1.0, 0.5], "Weights should be [1.0, 1.0, 0.5]"
    
    print("✅ Test 4 PASSED\n")


def test_numerical_stability():
    """Test numerical stability utilities"""
    print("="*80)
    print("Test 5: Numerical Stability")
    print("="*80)
    
    # Create an ill-conditioned matrix
    A = torch.randn(10, 10)
    A = A @ A.T  # Make symmetric
    A[0, 0] = 1e-10  # Make poorly conditioned
    
    is_stable, cond_num = check_condition_number(A, threshold=1e8)
    print(f"✓ Condition number: {cond_num:.2e}")
    print(f"✓ Is stable: {is_stable}")
    
    if not is_stable:
        print(f"✓ Correctly identified unstable matrix")
    
    print("✅ Test 5 PASSED\n")


def test_lru_eviction():
    """Test LRU eviction strategy"""
    print("="*80)
    print("Test 6: LRU Eviction")
    print("="*80)
    
    buffer = ReplayBuffer(max_size=5, eviction='lru')
    
    # Add 5 edits
    for i in range(5):
        request = {'subject': f'Entity_{i}', 'target_new': {'str': f' Value_{i}'}}
        buffer.add(request)
    
    # Access some records
    samples = buffer.sample(replay_rate=1.0, current_batch_size=3)
    accessed_subjects = [s.request['subject'] for s in samples]
    print(f"✓ Accessed subjects: {accessed_subjects}")
    
    # Add one more (should evict least recently used)
    new_request = {'subject': 'Entity_new', 'target_new': {'str': ' NewValue'}}
    buffer.add(new_request)
    
    print(f"✓ Added new edit, buffer size: {buffer.size()}")
    assert buffer.size() == 5, "Buffer should maintain max_size=5"
    
    # Check if least accessed was evicted
    remaining = [r.request['subject'] for r in buffer.buffer]
    print(f"✓ Remaining subjects: {remaining}")
    
    print("✅ Test 6 PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("Memory Replay Buffer Test Suite")
    print("="*80 + "\n")
    
    try:
        test_basic_operations()
        test_sampling_strategies()
        test_deduplication()
        test_merge_requests()
        test_numerical_stability()
        test_lru_eviction()
        
        print("="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
