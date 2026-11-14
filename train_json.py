"""
Standalone End-to-End Training Script: LoopedCartanLM on ChristianAzinn/json-training
- 2 unique CartanBlocks alternated K=10 times (~350k-450k params, ~1.5M+ effective unrolled compute)
- Zero external repo dependencies: runs standalone on Colab A100/L4/T4 or local GPU/CPU.
"""
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset


# =============================================================================
# 1. Standalone Cartan 2x2 Symplectic-Unitary Components
# =============================================================================

def chunked_log_scan_real_split(u_r, u_i, alpha, theta, chunk_size=64):
    """FLA-style real-split chunked log-cumsum causal scan: h_t = exp(-a_t + i*theta_t) h_{t-1} + u_t."""
    B, H, T, N = u_r.shape
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u_r = F.pad(u_r, (0, 0, 0, pad))
        u_i = F.pad(u_i, (0, 0, 0, pad))
        alpha = F.pad(alpha, (0, 0, 0, pad))
        theta = F.pad(theta, (0, 0, 0, pad))
    T_pad = u_r.shape[2]
    C = T_pad // chunk_size

    u_r_c = u_r.view(B, H, C, chunk_size, N)
    u_i_c = u_i.view(B, H, C, chunk_size, N)
    a_c = alpha.view(B, H, C, chunk_size, N)
    t_c = theta.view(B, H, C, chunk_size, N)

    g_a = torch.cumsum(-a_c, dim=3)
    g_t = torch.cumsum(t_c, dim=3)
    g_a_rel = g_a - g_a[:, :, :, :1, :]

    exp_neg = torch.exp(torch.clamp(-g_a_rel, max=20.0))
    cos_t, sin_t = torch.cos(g_t), torch.sin(g_t)
    w_r = exp_neg * (u_r_c * cos_t + u_i_c * sin_t)
    w_i = exp_neg * (-u_r_c * sin_t + u_i_c * cos_t)

    s_r = torch.cumsum(w_r, dim=3)
    s_i = torch.cumsum(w_i, dim=3)

    exp_pos = torch.exp(torch.clamp(g_a_rel, max=20.0))
    h_intra_r = exp_pos * (s_r * cos_t - s_i * sin_t)
    h_intra_i = exp_pos * (s_r * sin_t + s_i * cos_t)

    # Inter-chunk boundary propagation
    h_r = torch.zeros_like(h_intra_r)
    h_i = torch.zeros_like(h_intra_i)
    carry_r = torch.zeros(B, H, N, device=u_r.device, dtype=u_r.dtype)
    carry_i = torch.zeros(B, H, N, device=u_r.device, dtype=u_r.dtype)

    for c in range(C):
        ga_c = g_a[:, :, c]
        gt_c = g_t[:, :, c]
        exp_c = torch.exp(torch.clamp(ga_c, max=20.0))
        cc, sc = torch.cos(gt_c), torch.sin(gt_c)
        prop_r = exp_c * (carry_r.unsqueeze(2) * cc - carry_i.unsqueeze(2) * sc)
        prop_i = exp_c * (carry_r.unsqueeze(2) * sc + carry_i.unsqueeze(2) * cc)
        cur_r = h_intra_r[:, :, c] + prop_r
        cur_i = h_intra_i[:, :, c] + prop_i
        h_r[:, :, c] = cur_r
        h_i[:, :, c] = cur_i
        carry_r = cur_r[:, :, -1]
        carry_i = cur_i[:, :, -1]

    h_r = h_r.view(B, H, T_pad, N)[:, :, :T, :]
    h_i = h_i.view(B, H, T_pad, N)[:, :, :T, :]
    return h_r, h_i


class CausalCartanSSM(nn.Module):
    def __init__(self, n_nodes, n_heads=8):
        super().__init__()
        assert n_nodes % n_heads == 0
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = n_nodes // n_heads
        self.alpha_proj = nn.Linear(n_nodes, n_nodes)
        self.theta_proj = nn.Linear(n_nodes, n_nodes)
        self.u_proj = nn.Linear(n_nodes, 2 * n_nodes)

    def forward(self, p):
        B, T, N = p.shape
        H, D = self.n_heads, self.head_dim
        alpha = F.softplus(self.alpha_proj(p)).view(B, T, H, D).transpose(1, 2)
        theta = self.theta_proj(p).view(B, T, H, D).transpose(1, 2)
        u_r, u_i = self.u_proj(p).chunk(2, dim=-1)
        u_r = u_r.view(B, T, H, D).transpose(1, 2)
        u_i = u_i.view(B, T, H, D).transpose(1, 2)
        h_r, _ = chunked_log_scan_real_split(u_r, u_i, alpha, theta)
        return h_r.transpose(1, 2).contiguous().view(B, T, N)


class MultiHeadCayleyRotor(nn.Module):
    def __init__(self, n_nodes, n_heads=8):
        super().__init__()
        assert n_nodes % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_nodes // n_heads
        self.A_r = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim) * 0.05)
        self.A_i = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim) * 0.05)

    def forward(self, q, p):
        B, T, N = q.shape
        H, D = self.n_heads, self.head_dim
        S_r = self.A_r - self.A_r.transpose(-1, -2)
        S_i = self.A_i + self.A_i.transpose(-1, -2)
        I = torch.eye(D, device=q.device, dtype=q.dtype).unsqueeze(0).expand(H, D, D)
        S = torch.complex(S_r.float(), S_i.float())
        Ic = torch.complex(I.float(), torch.zeros_like(I.float()))
        U = torch.linalg.solve(Ic + S, Ic - S)
        U_r, U_i = U.real.to(q.dtype), U.imag.to(q.dtype)

        qh = q.view(B * T, H, D).transpose(0, 1)
        ph = p.view(B * T, H, D).transpose(0, 1)
        q_out = torch.bmm(qh, U_r.transpose(-1, -2)) - torch.bmm(ph, U_i.transpose(-1, -2))
        p_out = torch.bmm(qh, U_i.transpose(-1, -2)) + torch.bmm(ph, U_r.transpose(-1, -2))
        return q_out.transpose(0, 1).contiguous().view(B, T, N), p_out.transpose(0, 1).contiguous().view(B, T, N)


class CartanBlock(nn.Module):
    def __init__(self, n_nodes, n_heads=8, dt=0.1):
        super().__init__()
        self.dt = dt
        self.chi = nn.Parameter(torch.randn(n_nodes) * 0.1)
        self.gamma = nn.Parameter(torch.randn(n_nodes) * 0.1)
        self.omega = nn.Parameter(torch.ones(n_nodes) + torch.randn(n_nodes) * 0.05)
        self.phi = nn.Parameter(torch.randn(n_nodes) * 0.05)
        self.ssm = CausalCartanSSM(n_nodes, n_heads)
        self.damp_gate = nn.Linear(n_nodes, n_nodes)
        self.rotor = MultiHeadCayleyRotor(n_nodes, n_heads)

    def _kick(self, q):
        return self.chi * torch.tanh(q) + self.gamma * torch.sin(self.omega * q + self.phi)

    def forward(self, q, p):
        p1 = p - 0.5 * self.dt * self._kick(q)
        q2 = q + self.dt * self.ssm(p1)
        gamma_q = F.softplus(self.damp_gate(q2))
        p2 = p1 * torch.exp(-gamma_q * self.dt)
        q3, p3 = self.rotor(q2, p2)
        p4 = p3 - 0.5 * self.dt * self._kick(q3)
        return q3, p4


class LoopedCartanLM(nn.Module):
    def __init__(self, vocab_size, n_nodes=128, n_heads=8, n_unique=2, loop_steps=10, dt=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.loop_steps = loop_steps
        self.tok_emb = nn.Embedding(vocab_size, 2 * n_nodes)
        self.blocks = nn.ModuleList([CartanBlock(n_nodes, n_heads, dt) for _ in range(n_unique)])
        self.norm = nn.LayerNorm(2 * n_nodes)
        self.head = nn.Linear(2 * n_nodes, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx)
        q, p = x[..., :self.n_nodes], x[..., self.n_nodes:]
        for k in range(self.loop_steps):
            q, p = self.blocks[k % len(self.blocks)](q, p)
        h = self.norm(torch.cat([q, p], dim=-1))
        return self.head(h)


# =============================================================================
# 2. Dataset & Training Loop (ChristianAzinn/json-training)
# =============================================================================

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = load_dataset("ChristianAzinn/json-training", split="train")
    corpus = "\n".join(
        f"<|q|>{row['query']}<|s|>{row['schema']}<|j|>{row['response']}<|e|>"
        for row in ds
    )
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)

    n_split = int(0.95 * len(data))
    train_data, val_data = data[:n_split], data[n_split:]
    print(f"Loaded ChristianAzinn/json-training | chars={len(data):,} | vocab={vocab_size}")

    BATCH_SIZE = 32
    SEQ_LEN = 256
    MAX_STEPS = 1000

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - SEQ_LEN - 1, (BATCH_SIZE,))
        x = torch.stack([src[i : i + SEQ_LEN] for i in ix]).to(device)
        y = torch.stack([src[i + 1 : i + 1 + SEQ_LEN] for i in ix]).to(device)
        return x, y

    model = LoopedCartanLM(
        vocab_size=vocab_size, n_nodes=128, n_heads=8, n_unique=2, loop_steps=10, dt=0.1
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LoopedCartanLM (2 unique blocks, K=10) | params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS, eta_min=1e-4)

    t0 = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        xb, yb = get_batch("train")
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch("val")
                v_loss = F.cross_entropy(model(xv).view(-1, vocab_size), yv.view(-1)).item()
            model.train()
            print(f"step {step:4d}/{MAX_STEPS} | train={loss.item():.4f} | val={v_loss:.4f} | ppl={math.exp(v_loss):.2f} | {time.perf_counter()-t0:.1f}s")

    # Quick generation check
    model.eval()
    prompt = '<|q|>Return a user object<|s|>{"name": "string", "age": "integer"}<|j|>'
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(120):
            logits = model(idx[:, -SEQ_LEN:])[:, -1, :]
            next_id = torch.multinomial(F.softmax(logits / 0.7, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if itos[next_id.item()] == "<":
                break
    out_text = "".join(itos[i] for i in idx[0].tolist())
    print("\n=== Sample Output ===\n", out_text)

    ckpt_path = "looped_cartan_json_261k.pt"
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
