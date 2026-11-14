"""
Generate crisp white-background publication visuals for LoopedCartanLM.
"""
import os
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0f172a",
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "font.size": 10,
})


def make_pareto_chart():
    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=220)
    pts = [
        ("LoopedCartanLM (34k, T=128)", 34244, 9.55, "#059669", "o", 130, (10, 6)),
        ("UnrolledCartanLM (111k, T=128)", 111632, 9.21, "#0d9488", "s", 110, (10, -12)),
        ("CartanLM (60k, T=64)", 60040, 8.01, "#0284c7", "^", 125, (10, -10)),
        ("Causal GPT (60k, T=64)", 60004, 12.92, "#dc2626", "X", 125, (10, 4)),
        ("Causal GPT (3.31M, T=512)", 3307264, 13.35, "#b91c1c", "X", 140, (-145, -16)),
        ("CartanLM (1.50M, T=512)", 1495056, 7.19, "#2563eb", "D", 130, (-135, 8)),
    ]
    for label, x, y, c, m, s, offset in pts:
        ax.scatter(x, y, c=c, marker=m, s=s, zorder=5, edgecolors="#0f172a", linewidths=0.7)
        ax.annotate(
            f"{label}\n({y:.2f} PPL)", (x, y), xytext=offset, textcoords="offset points",
            fontsize=8.2, color="#1e293b", fontweight="medium"
        )
    ax.set_xscale("log")
    ax.set_ylim(6.0, 14.8)
    ax.set_xlabel("Parameter Count (log scale)", fontweight="bold")
    ax.set_ylabel("Validation Perplexity (PPL, lower is better)", fontweight="bold")
    ax.set_title("TinyShakespeare Parameter-Efficiency Pareto Frontier", fontweight="bold", pad=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("assets/looped_cartan_pareto.png")
    plt.close()


def make_bch_and_erank_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=220)

    labels = ["1-Block Loop\n(62 params)", "2-Block Alt Loop\n(92 params)", "Unrolled L=12\n(392 params)"]
    accs = [95.00, 99.00, 100.00]
    colors = ["#d97706", "#059669", "#2563eb"]
    bars = ax1.bar(labels, accs, color=colors, edgecolor="#0f172a", linewidth=0.7, width=0.52)
    for b, v in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", color="#0f172a")
    ax1.set_ylim(88, 102.5)
    ax1.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax1.set_title("BCH Bracket Advantage on Double Spiral (K=12)", fontweight="bold")
    ax1.grid(True, axis="y")

    ts = [32, 64, 128]
    looped_erank = [22.03, 22.11, 21.56]
    unrolled_erank = [22.42, 21.33, 19.65]
    ax2.plot(ts, looped_erank, marker="o", linewidth=2.4, color="#059669", label="LoopedCartanLM (34k, 2x4)")
    ax2.plot(ts, unrolled_erank, marker="s", linewidth=2.4, linestyle="--", color="#2563eb", label="UnrolledCartanLM (111k, L=8)")
    ax2.set_xticks(ts)
    ax2.set_xlabel("Sequence Position t", fontweight="bold")
    ax2.set_ylabel("Roy-Vetterli Effective Rank (eRank)", fontweight="bold")
    ax2.set_title("Temporal eRank Retention (TinyShakespeare, T=128)", fontweight="bold")
    ax2.set_ylim(18.5, 23.5)
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/bch_loop_ablation.png")
    plt.close()


def make_json_training_chart():
    steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
             2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000,
             4200, 4400, 4600, 4800, 5000]
    val_loss = [2.6440, 1.5206, 1.5290, 1.3586, 1.2796, 1.0967, 1.1158, 1.0735, 0.9646, 1.1862,
                0.8659, 0.9478, 0.9985, 0.9895, 0.8525, 0.8520, 0.8968, 1.1335, 0.9784, 1.0174,
                0.6979, 0.9372, 0.9633, 0.8594, 0.7794]
    ppl = [14.07, 4.57, 4.61, 3.89, 3.60, 2.99, 3.05, 2.93, 2.62, 3.27,
           2.38, 2.58, 2.71, 2.69, 2.35, 2.34, 2.45, 3.11, 2.66, 2.77,
           2.01, 2.55, 2.62, 2.36, 2.18]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=220)
    ax1.plot(steps, val_loss, marker="o", markersize=4.5, linewidth=2.2, color="#0284c7")
    ax1.set_xlabel("Training Step", fontweight="bold")
    ax1.set_ylabel("Validation Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("261k LoopedCartanLM on ChristianAzinn/json-training", fontweight="bold")
    ax1.grid(True)

    ax2.plot(steps, ppl, marker="o", markersize=4.5, linewidth=2.2, color="#059669")
    ax2.axhline(2.01, color="#d97706", linestyle=":", linewidth=1.6, label="Best Val PPL: 2.01 (step 4200)")
    ax2.set_xlabel("Training Step", fontweight="bold")
    ax2.set_ylabel("Validation Perplexity (PPL)", fontweight="bold")
    ax2.set_title("JSON Schema Perplexity Convergence (869.3s CUDA)", fontweight="bold")
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/json_training_curve.png")
    plt.close()


if __name__ == "__main__":
    make_pareto_chart()
    make_bch_and_erank_chart()
    make_json_training_chart()
    print("Generated all 3 white-background charts in assets/")
