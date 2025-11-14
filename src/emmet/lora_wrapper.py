"""
Minimal LoRA Wrapper for EMMET

This module provides a lightweight LoRA (Low-Rank Adaptation) wrapper that can be
applied after EMMET editing to add trainable low-rank adjustments.

Design Philosophy:
- Post-processing approach: Apply LoRA AFTER EMMET editing
- Minimal modification: Does not change EMMET's closed-form solution
- Parameter efficient: Adds only r*(d_in + d_out) trainable parameters per layer

Usage:
    from emmet.lora_wrapper import apply_lora_to_edited_model
    
    # After EMMET editing
    edited_model = apply_emmet_to_model(model, tok, requests, hparams)
    
    # Add LoRA adaptation
    lora_model = apply_lora_to_edited_model(
        edited_model,
        target_modules=['mlp.c_fc', 'mlp.c_proj'],
        rank=8,
        alpha=16
    )
"""

import logging
from typing import Dict, List, Optional, Set
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Single LoRA layer implementing W' = W + (alpha/r) * B @ A
    
    Args:
        original_weight: The base weight matrix (after EMMET editing)
        rank: Rank of low-rank decomposition
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        original_weight: torch.Tensor,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Store original weight (frozen)
        self.register_buffer('base_weight', original_weight.detach().clone())
        
        out_features, in_features = original_weight.shape
        
        # Initialize LoRA matrices
        # A: (rank, in_features) - initialized with kaiming uniform
        # B: (out_features, rank) - initialized with zeros (stable init)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # B starts at zero so initially LoRA has no effect
        nn.init.zeros_(self.lora_B)
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        
        # Track whether LoRA is enabled
        self.lora_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = (W_base + scaling * B @ A) @ x
        """
        # Base transformation
        # Include bias if present (kept as attribute by wrapper)
        bias = getattr(self, "bias", None)
        result = torch.nn.functional.linear(x, self.base_weight, bias)
        
        # Add LoRA adjustment if enabled
        if self.lora_enabled and self.rank > 0:
            lora_x = x
            if self.dropout is not None:
                lora_x = self.dropout(lora_x)
            
            # LoRA path: x @ A^T @ B^T
            lora_out = torch.nn.functional.linear(lora_x, self.lora_A)
            lora_out = torch.nn.functional.linear(lora_out, self.lora_B)
            result = result + self.scaling * lora_out
        
        return result
    
    def enable_lora(self):
        """Enable LoRA adjustments"""
        self.lora_enabled = True
    
    def disable_lora(self):
        """Disable LoRA adjustments (use only base weight)"""
        self.lora_enabled = False
    
    def merge_lora(self) -> torch.Tensor:
        """
        Merge LoRA into base weight: W_merged = W_base + scaling * B @ A
        Returns the merged weight tensor.
        """
        if self.rank > 0:
            delta = self.scaling * (self.lora_B @ self.lora_A)
            return self.base_weight + delta
        return self.base_weight.clone()
    
    def get_lora_params(self) -> int:
        """Return number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()


class MinimalLoRAWrapper:
    """
    Minimal LoRA wrapper for EMMET-edited models
    
    This wrapper adds LoRA layers to specified modules after EMMET editing.
    It maintains a registry of LoRA layers and provides utilities for training,
    merging, and parameter management.
    
    Args:
        model: The model after EMMET editing
        target_modules: List of module names to add LoRA (e.g., ['mlp.c_fc', 'attn.c_attn'])
        rank: Rank for all LoRA layers
        alpha: Alpha scaling for all LoRA layers
        dropout: Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        target_modules: List[str],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Registry of LoRA layers: {full_module_path: LoRALayer}
        self.lora_layers: Dict[str, LoRALayer] = {}
        
        # Apply LoRA to target modules
        self._apply_lora()
        
        logger.info(f"Applied LoRA (rank={rank}, alpha={alpha}) to {len(self.lora_layers)} modules")
    
    def _apply_lora(self):
        """Apply LoRA to all target modules in the model"""
        for name, module in self.model.named_modules():
            # Check if this module matches any target pattern
            if self._is_target_module(name):
                if isinstance(module, nn.Linear):
                    self._replace_with_lora(name, module)
    
    def _is_target_module(self, module_name: str) -> bool:
        """Check if module name matches any target pattern"""
        for target in self.target_modules:
            if target in module_name:
                return True
        return False
    
    def _replace_with_lora(self, module_path: str, linear_module: nn.Linear):
        """Replace a Linear module with LoRA-enhanced version"""
        # Create LoRA layer
        lora_layer = LoRALayer(
            original_weight=linear_module.weight.data,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        # Copy bias if present
        if linear_module.bias is not None:
            lora_layer.bias = nn.Parameter(linear_module.bias.data.clone())
        else:
            lora_layer.bias = None
        
        # Replace module in model
        parent_name, child_name = self._split_module_path(module_path)
        parent_module = self._get_module_by_path(self.model, parent_name)
        setattr(parent_module, child_name, lora_layer)
        
        # Register in our tracking dict
        self.lora_layers[module_path] = lora_layer
        
        logger.debug(f"Replaced {module_path} with LoRA layer (rank={self.rank})")
    
    def _split_module_path(self, path: str) -> tuple:
        """Split 'a.b.c' into ('a.b', 'c')"""
        parts = path.split('.')
        if len(parts) == 1:
            return '', parts[0]
        return '.'.join(parts[:-1]), parts[-1]
    
    def _get_module_by_path(self, model, path: str):
        """Get module by dotted path"""
        if not path:
            return model
        module = model
        for part in path.split('.'):
            module = getattr(module, part)
        return module
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters"""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def freeze_base_model(self):
        """Freeze all non-LoRA parameters"""
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        logger.info("Froze all base model parameters")
    
    def enable_lora(self):
        """Enable all LoRA layers"""
        for lora_layer in self.lora_layers.values():
            lora_layer.enable_lora()
    
    def disable_lora(self):
        """Disable all LoRA layers (use only base weights)"""
        for lora_layer in self.lora_layers.values():
            lora_layer.disable_lora()
    
    def merge_lora(self):
        """Merge all LoRA weights into base weights"""
        for module_path, lora_layer in self.lora_layers.items():
            merged_weight = lora_layer.merge_lora()
            
            # Replace LoRA layer with standard Linear layer
            parent_name, child_name = self._split_module_path(module_path)
            parent_module = self._get_module_by_path(self.model, parent_name)
            
            out_features, in_features = merged_weight.shape
            new_linear = nn.Linear(in_features, out_features, bias=(lora_layer.bias is not None))
            new_linear.weight.data = merged_weight
            if lora_layer.bias is not None:
                new_linear.bias.data = lora_layer.bias.data
            
            setattr(parent_module, child_name, new_linear)
        
        self.lora_layers.clear()
        logger.info("Merged all LoRA weights into base model")
    
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params = sum(lora.get_lora_params() for lora in self.lora_layers.values())
        base_params = total_params - lora_params
        
        return {
            'total': total_params,
            'base': base_params,
            'lora': lora_params,
            'lora_percentage': 100.0 * lora_params / total_params if total_params > 0 else 0
        }
    
    def print_param_stats(self):
        """Print parameter statistics"""
        stats = self.get_param_count()
        logger.info("="*60)
        logger.info("LoRA Parameter Statistics:")
        logger.info(f"  Total parameters:      {stats['total']:,}")
        logger.info(f"  Base parameters:       {stats['base']:,}")
        logger.info(f"  LoRA parameters:       {stats['lora']:,}")
        logger.info(f"  LoRA percentage:       {stats['lora_percentage']:.2f}%")
        logger.info(f"  Number of LoRA layers: {len(self.lora_layers)}")
        logger.info("="*60)


def apply_lora_to_edited_model(
    model: AutoModelForCausalLM,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True
) -> MinimalLoRAWrapper:
    """
    Apply LoRA to an EMMET-edited model
    
    Args:
        model: Model after EMMET editing
        target_modules: List of module name patterns to apply LoRA
                       Default: ['mlp.c_fc', 'mlp.c_proj'] for GPT-2 style models
        rank: Rank of LoRA decomposition
        alpha: Scaling factor for LoRA
        dropout: Dropout probability for LoRA path
        freeze_base: Whether to freeze non-LoRA parameters
    
    Returns:
        MinimalLoRAWrapper instance managing the LoRA-enhanced model
    
    Example:
        >>> # After EMMET editing
        >>> edited_model = apply_emmet_to_model(model, tok, requests, hparams)
        >>> 
        >>> # Add LoRA
        >>> lora_wrapper = apply_lora_to_edited_model(
        ...     edited_model,
        ...     target_modules=['mlp.c_fc', 'mlp.c_proj'],
        ...     rank=8
        ... )
        >>> 
        >>> # Get trainable parameters for optimizer
        >>> optimizer = torch.optim.Adam(lora_wrapper.get_trainable_params(), lr=1e-4)
        >>> 
        >>> # After training, merge LoRA back
        >>> lora_wrapper.merge_lora()
    """
    if target_modules is None:
        # Default targets for GPT-2 style models
        target_modules = ['mlp.c_fc', 'mlp.c_proj']
    
    wrapper = MinimalLoRAWrapper(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    if freeze_base:
        wrapper.freeze_base_model()
    
    wrapper.print_param_stats()
    
    return wrapper


def get_lora_target_modules(model_name: str) -> List[str]:
    """
    Get recommended LoRA target modules for different model architectures
    
    Args:
        model_name: Model name or type (e.g., 'gpt2', 'llama', 'gpt-j')
    
    Returns:
        List of recommended target module patterns
    """
    model_name = model_name.lower()
    
    if 'gpt2' in model_name or 'gpt-2' in model_name:
        # GPT-2: Apply to MLP layers
        return ['mlp.c_fc', 'mlp.c_proj']
    
    elif 'llama' in model_name:
        # LLaMA: Apply to MLP layers
        return ['mlp.up_proj', 'mlp.down_proj', 'mlp.gate_proj']
    
    elif 'gpt-j' in model_name or 'gptj' in model_name:
        # GPT-J: Apply to MLP layers
        return ['mlp.fc_in', 'mlp.fc_out']
    
    elif 'opt' in model_name:
        # OPT: Apply to MLP layers
        return ['fc1', 'fc2']
    
    else:
        # Generic fallback
        logger.warning(f"Unknown model type '{model_name}', using generic targets")
        return ['mlp', 'fc']
