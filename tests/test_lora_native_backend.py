import torch
import torch.nn as nn

from emmet.peft_backend_lora import LoRANativeBackend


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.norm(a)
    if denom.item() == 0:
        return float(torch.norm(b))
    return float(torch.norm(a - b) / denom)


class Toy(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        return self.lin(x)


def test_svd_projection_precision():
    torch.manual_seed(0)
    d_in, d_out = 64, 48
    model = Toy(d_in, d_out, bias=False)
    delta = torch.randn(d_out, d_in)

    w_before = model.lin.weight.detach().clone()

    be = LoRANativeBackend(model, rank=16, alpha=16.0, scale=1.0, freeze_base=True)
    be.apply_delta('lin.weight', delta, use_svd=True, fit_steps=0)

    layer = be.get_lora_layer('lin')
    approx = layer.lora_B @ layer.lora_A
    err = rel_fro(delta, approx)
    assert err <= 1e-4, f"SVD mapping relative error too high: {err}"

    # Base weight unchanged in native mode
    assert torch.allclose(layer.base_weight, w_before), "Base weight must remain frozen"


def test_clear_and_fallback_to_raw():
    torch.manual_seed(0)
    d_in, d_out = 32, 24
    model = Toy(d_in, d_out, bias=False)

    delta = torch.randn(d_out, d_in)
    be = LoRANativeBackend(model, rank=4, alpha=4.0, scale=1.0, freeze_base=True)
    be.apply_delta('lin.weight', delta, use_svd=True, fit_steps=0)

    # zero factors
    be.clear_delta('lin')
    layer = be.get_lora_layer('lin')
    assert torch.norm(layer.lora_B @ layer.lora_A).item() == 0.0

    # fallback to raw should replace module with nn.Linear
    be.fallback_to_raw('lin.weight', delta)
    assert isinstance(model.lin, nn.Linear)
    # and weight should match base + delta approximately
    expected = layer.base_weight + delta
    assert torch.allclose(model.lin.weight.detach(), expected, atol=1e-6)
