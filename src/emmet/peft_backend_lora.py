"""
LoRA Native Backend for EMMET

This backend embeds LoRA directly into the EMMET editing process by mapping the
closed-form update matrix ΔW at each edited layer into low-rank LoRA factors:
ΔW ≈ (alpha/r) * B @ A, with A ∈ R^{r×in}, B ∈ R^{out×r}.

Design goals:
- Keep base weights frozen; apply edits through LoRA overlays only.
- SVD-based closed-form projection with optional tiny fitting.
- Minimal intrusion: reuse existing LoRALayer from lora_wrapper.

Notes:
- By default, we set alpha = rank so that scaling = 1.0, and store the
  entire ΔW in B @ A (optionally multiplied by `scale`).
- If your shapes are transposed by model family, ensure ΔW matches weight
  shape before calling `apply_delta` (callers should use upd_matrix_match_shape).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .lora_wrapper import LoRALayer


class LoRANativeBackend:
    """Manage LoRA overlays for native EMMET integration."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: Optional[float] = None,
        scale: float = 1.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        self.model = model
        self.rank = int(rank)
        # Default alpha so that scaling = 1.0 in LoRALayer
        self.alpha = float(alpha if alpha is not None else self.rank)
        self.scale = float(scale)
        self.dropout = float(dropout)
        self._registry: Dict[str, LoRALayer] = {}

        if freeze_base:
            for n, p in self.model.named_parameters():
                p.requires_grad = False

    def _split_module_path(self, path: str):
        parts = path.split(".")
        if len(parts) == 1:
            return "", parts[0]
        return ".".join(parts[:-1]), parts[-1]

    def _get_module_by_path(self, root: nn.Module, path: str) -> nn.Module:
        if not path:
            return root
        mod = root
        for part in path.split("."):
            mod = getattr(mod, part)
        return mod

    def _ensure_lora_layer(self, module_path: str) -> LoRALayer:
        """Replace a Linear module at `module_path` with a LoRALayer if not already."""
        if module_path in self._registry:
            return self._registry[module_path]

        parent_name, child_name = self._split_module_path(module_path)
        parent = self._get_module_by_path(self.model, parent_name)
        orig = getattr(parent, child_name)
        if not isinstance(orig, nn.Linear):
            raise TypeError(f"Module at {module_path} is not nn.Linear: {type(orig)}")

        lora = LoRALayer(
            original_weight=orig.weight.data,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )
        # copy bias if exists
        if orig.bias is not None:
            lora.bias = nn.Parameter(orig.bias.data.clone(), requires_grad=False)
        else:
            lora.bias = None

        setattr(parent, child_name, lora)
        self._registry[module_path] = lora
        return lora

    def get_lora_layer(self, module_path: str) -> LoRALayer:
        """Return the LoRA layer for a given module path (without .weight)."""
        return self._registry[module_path]

    @torch.no_grad()
    def clear_delta(self, module_path: str) -> None:
        """Zero out LoRA factors for a given module path (without .weight)."""
        layer = self._registry.get(module_path)
        if layer is not None:
            layer.lora_A.zero_()
            layer.lora_B.zero_()

    @torch.no_grad()
    def apply_delta(
        self,
        weight_param_name: str,
        delta: torch.Tensor,
        use_svd: bool = True,
        fit_steps: int = 0,
        fit_lr: float = 1e-2,
    ) -> None:
        """
        Apply a delta matrix to the linear layer specified by its parameter name
        (e.g., 'transformer.h.10.mlp.c_fc.weight').

        Replaces that layer with LoRALayer and sets lora_A, lora_B so that
        (alpha/r) * B @ A ≈ delta * scale.
        """
        assert weight_param_name.endswith(".weight"), "Expected a .weight parameter name"
        module_path = weight_param_name[:-7]  # strip '.weight'

        # Ensure LoRA layer exists
        lora = self._ensure_lora_layer(module_path)

        # Target matrix to approximate
        target = delta.detach().to(lora.base_weight.device).clone() * self.scale

        if use_svd:
            # Full or economic SVD depending on dims
            U, S, Vh = torch.linalg.svd(target, full_matrices=False)
            r = min(self.rank, S.numel())
            if r == 0:
                # nothing to do
                lora.lora_A.zero_()
                lora.lora_B.zero_()
                return
            Sr = S[:r]
            Ur = U[:, :r]
            Vhr = Vh[:r, :]
            # Construct factors so that B @ A ≈ target
            # B: (out, r) = Ur * sqrt(Sr)
            # A: (r, in)  = sqrt(Sr) * Vhr
            sqrtSr = torch.sqrt(Sr)
            B = Ur * sqrtSr.unsqueeze(0)
            A = sqrtSr.unsqueeze(1) * Vhr

            # Set parameters
            lora.lora_A.data.zero_()
            lora.lora_B.data.zero_()
            # Fit into existing parameter sizes (rank may be > r)
            lora.lora_A.data[:r, :A.shape[1]] = A
            lora.lora_B.data[:B.shape[0], :r] = B
        else:
            # Initialize with zeros then do tiny gradient fit to minimize ||B@A - target||_F
            # Use current rank sizes
            lora.lora_A.data.zero_()
            lora.lora_B.data.zero_()

        # Optional tiny fitting to refine
        if fit_steps > 0:
            lora.lora_A.requires_grad_(True)
            lora.lora_B.requires_grad_(True)
            opt = torch.optim.Adam([lora.lora_A, lora.lora_B], lr=fit_lr)
            for _ in range(int(fit_steps)):
                opt.zero_grad(set_to_none=True)
                approx = lora.lora_B @ lora.lora_A
                loss = torch.nn.functional.mse_loss(approx, target)
                loss.backward()
                opt.step()
            lora.lora_A.requires_grad_(False)
            lora.lora_B.requires_grad_(False)

        # Return nothing; callers can compute residual from registry if needed
        # Return nothing; callers can compute residual from registry if needed

    @torch.no_grad()
    def fallback_to_raw(self, weight_param_name: str, delta: torch.Tensor) -> None:
        """Replace LoRA layer (if present) with nn.Linear and apply raw delta to base weight.

        weight_param_name: full parameter name ending with .weight
        delta: matrix shaped like target weight
        """
        assert weight_param_name.endswith(".weight"), "Expected a .weight parameter name"
        module_path = weight_param_name[:-7]

        parent_name, child_name = self._split_module_path(module_path)
        parent = self._get_module_by_path(self.model, parent_name)
        current = getattr(parent, child_name)

        # Determine base weight and bias
        if isinstance(current, LoRALayer):
            base_w = current.base_weight.detach().clone()
            bias = getattr(current, 'bias', None)
        elif isinstance(current, nn.Linear):
            base_w = current.weight.detach().clone()
            bias = current.bias.detach().clone() if current.bias is not None else None
        else:
            raise TypeError(f"Unsupported module type for fallback: {type(current)}")

        merged = base_w + delta.to(base_w.device)
        out_features, in_features = merged.shape
        new_linear = nn.Linear(in_features, out_features, bias=(bias is not None))
        new_linear.weight.data.copy_(merged)
        if bias is not None:
            new_linear.bias.data.copy_(bias)

        setattr(parent, child_name, new_linear)
        # remove registry entry if existed
        if module_path in self._registry:
            del self._registry[module_path]

    def stats(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.model.parameters())
        lora_params = 0
        for layer in self._registry.values():
            lora_params += layer.lora_A.numel() + layer.lora_B.numel()
        return {
            "total_params": float(total),
            "lora_params": float(lora_params),
            "lora_percentage": float(lora_params) / float(total) * 100.0 if total > 0 else 0.0,
        }
