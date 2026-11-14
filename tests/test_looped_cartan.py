import torch
from looped_cartan import LoopedCartanLM, MultiHeadCayleyRotor


def test_forward_shape_and_weight_tying():
    model = LoopedCartanLM(vocab_size=65, n_nodes=32, n_heads=4, n_unique=2, loop_steps=6)
    x = torch.randint(0, 65, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 65)
    assert model.head.weight is model.tok_emb.weight


def test_cayley_rotor_unitarity():
    rotor = MultiHeadCayleyRotor(n_nodes=32, n_heads=4)
    q = torch.randn(2, 8, 32)
    p = torch.randn(2, 8, 32)
    q_out, p_out = rotor(q, p)
    norm_in = (q ** 2 + p ** 2).sum(dim=-1)
    norm_out = (q_out ** 2 + p_out ** 2).sum(dim=-1)
    assert torch.allclose(norm_in, norm_out, atol=1e-5)


def test_strict_causality():
    model = LoopedCartanLM(vocab_size=65, n_nodes=32, n_heads=4, n_unique=2, loop_steps=4)
    x = torch.randint(0, 65, (1, 12))
    emb = model.tok_emb(x).detach().requires_grad_(True)
    q, p = emb[..., :32], emb[..., 32:]
    for k in range(model.loop_steps):
        q, p = model.blocks[k % 2](q, p)
    loss = q[:, 5, :].sum()
    loss.backward()
    future_grad = emb.grad[:, 6:, :].abs().max().item()
    assert future_grad == 0.0
