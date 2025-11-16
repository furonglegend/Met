from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from utils.hparams import HyperParams


@dataclass
class EMMETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    #Objective
    calculate_objective_value: bool
    update_norm_lambda: float
    emmet_lambda: float

    # LoRA-native integration (optional; defaults keep raw editing)
    edit_mode: Literal["raw", "lora_native"] = "raw"
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_scale: float = 1.0
    lora_use_svd: bool = True
    lora_fit_steps: int = 0
    allow_fallback: bool = True
    lora_residual_threshold: Optional[float] = 0.3

    # Trust / Rollback (Phase 4)
    trust_enable: bool = False
    trust_threshold: float = 0.3
    trust_action: Literal["rollback", "scale"] = "rollback"
    trust_scale: float = 0.5
    trust_heldout_samples: int = 0
    # Optional component weights, e.g., {"gain":0.7, "preserve":0.3}
    trust_weights: Optional[Dict[str, float]] = None
