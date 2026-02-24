from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------
# Experiment configuration
# ------------------------------
@dataclass
class Config:
    num_heads: int = 4
    hidden_dim: int = 64  
    n: int = 10  # grid is n x n
    train_epochs: int = 64  # E
    num_trials: int = 5  # M

    # Training / model settings
    num_layers: int = 2          # Number of transformer encoder layers. 
    ff_mult: int = 4             # Feed-forward hidden dim = hidden_dim * ff_mult.
    dropout: float = 0.1         
    batch_size: int = 32        
    lr: float = 3e-4            # Learning rate for AdamW (transformer).
    baseline_lr: float = 5e-4   # Learning rate for AdamW (linear baseline)
    lr_decay_epoch: int = 20    # After this epoch, both LRs are set to lr_after_decay / baseline_lr_after_decay.
    lr_after_decay: float = 1e-4       # Transformer LR after lr_decay_epoch.
    baseline_lr_after_decay: float = 2e-4  # Baseline LR after lr_decay_epoch.
    train_sequences: int = 1200 # Number of random-walk sequences used for training.
    walk_length: int = 128      # Steps per random walk.
    max_seq_len: int = 256      # Maximum sequence length the model supports (position embeddings, causal mask).
    cap_eval_at_train_length: bool = True  # If True, never run model with more than walk_length tokens.

    # Evaluation settings 
    prompt_len: int = 32       # Length of the fixed input prompt used to start generation (first prompt_len tokens of a random walk).
    generate_steps: int = 80   # S: number of tokens generated per trial; we record logits at each step → S×D per trial.
    max_lag: int = 96          # m: maximum lag tau; we compute time-lagged covariance for tau = 1, 2, ..., m (capped by S-1 at runtime).
    temperatures: Tuple[float, ...] = (0.01, 0.5, 0.7, 1.0, 1.3)  # T>0: sampling temperatures
    eval_epochs: Tuple[int, ...] = ()  # If empty, set in main to (0, 2, 4, 8, 16, ...) capped by train_epochs.

    # Misc
    seed: int = 42
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grid_to_token(r: int, c: int, n: int) -> int:
    return r * n + c


def random_walk_sequence(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create one random-walk token sequence on an n x n grid.
    Returns length+1 tokens so that x=[:-1], y=[1:].
    """
    r = int(rng.integers(0, n))
    c = int(rng.integers(0, n))
    tokens = [grid_to_token(r, c, n)]

    # 4-neighbor random walk with wrap-around.
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(length):
        dr, dc = moves[int(rng.integers(0, 4))]
        r = (r + dr) % n
        c = (c + dc) % n
        tokens.append(grid_to_token(r, c, n))
    return np.asarray(tokens, dtype=np.int64)


def build_dataset(cfg: Config, rng: np.random.Generator) -> TensorDataset:
    seqs = [random_walk_sequence(cfg.n, cfg.walk_length, rng) for _ in range(cfg.train_sequences)]
    arr = np.stack(seqs, axis=0)  # [N, L+1]
    x = torch.from_numpy(arr[:, :-1])  # [N, L]
    y = torch.from_numpy(arr[:, 1:])  # [N, L]
    return TensorDataset(x, y)


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, hidden_dim: int, num_heads: int, num_layers: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pos_emb.shape[1]}")

        x = self.token_emb(idx) + self.pos_emb[:, :seq_len, :]

        # Causal mask: prevent each position from attending to future positions.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, L, V]
        return logits


class LinearBaseline(nn.Module):
    """
    Linear dynamical baseline (no process noise):
      x_{t+1} = A x_t + B e(token_t)
      logits_t = C x_t
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        latent_dim: int,
        input_dim: int,
        spectral_radius: float = 0.98,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Token embedding used as the linear input u_t.
        self.token_emb = nn.Embedding(vocab_size, input_dim)
        self.B = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.05)
        self.C = nn.Parameter(torch.randn(vocab_size, latent_dim) * 0.05)

        A = torch.randn(latent_dim, latent_dim) / np.sqrt(max(1, latent_dim))
        # Scale A to desired spectral radius for "stable" dynamics.
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(A).abs()
            max_eig = float(torch.max(eigvals).item())
            if max_eig < 1e-8:
                max_eig = 1.0
            A = A * (spectral_radius / max_eig)
        self.A = nn.Parameter(A)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def consume_tokens(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        x = state
        for t in range(tokens.shape[1]):
            u = self.token_emb(tokens[:, t])  # [B, input_dim]
            x = x @ self.A.T + u @ self.B.T
        return x

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.C.T  # [B, V]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Compatibility method (shape mirrors transformer forward output).
        bsz, seq_len = idx.shape
        x = self.init_state(bsz, idx.device)
        outs: List[torch.Tensor] = []
        for t in range(seq_len):
            x = self.consume_tokens(x, idx[:, t : t + 1])
            outs.append(self.logits_from_state(x).unsqueeze(1))
        return torch.cat(outs, dim=1)


@torch.no_grad()
def collect_trial_logits(
    model: nn.Module,
    prompt: torch.Tensor,
    *,
    temperature: float,
    num_trials: int,
    steps: int,
    vocab_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Return L with shape [M, S, D], where:
    - M = num_trials
    - S = generated steps
    - D = vocab_size
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0.")

    model.eval()
    all_trials: List[np.ndarray] = []

    for _ in range(num_trials):
        seq = prompt.clone().to(device)  # [1, P]
        trial_logits: List[np.ndarray] = []

        for _step in range(steps):
            logits = model(seq)[:, -1, :]  # [1, D]
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1,1]

            trial_logits.append(logits.squeeze(0).detach().cpu().numpy())
            seq = torch.cat([seq, next_tok], dim=1)

        all_trials.append(np.stack(trial_logits, axis=0))

    out = np.stack(all_trials, axis=0)
    assert out.shape == (num_trials, steps, vocab_size)
    return out


@torch.no_grad()
def collect_trial_logits_linear(
    model: LinearBaseline,
    prompt: torch.Tensor,
    *,
    temperature: float,
    num_trials: int,
    steps: int,
    vocab_size: int,
    device: torch.device,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0.")

    model.eval()
    all_trials: List[np.ndarray] = []
    prompt = prompt.to(device)

    for _ in range(num_trials):
        state = model.init_state(batch_size=1, device=device)
        state = model.consume_tokens(state, prompt)
        trial_logits: List[np.ndarray] = []

        for _step in range(steps):
            logits = model.logits_from_state(state)  # [1, D]
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1,1]

            trial_logits.append(logits.squeeze(0).detach().cpu().numpy())
            state = model.consume_tokens(state, next_tok)

        all_trials.append(np.stack(trial_logits, axis=0))

    out = np.stack(all_trials, axis=0)
    assert out.shape == (num_trials, steps, vocab_size)
    return out


def time_lagged_covariance(L: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute lagged covariance for vector time series from trial logits.
    L shape: [M, S, D].

    Uses corrected indexing and per-step centering across trials:
      mu_s = mean_i L[i,s]
      C_tau = 1 / (M * (S - tau)) * sum_i sum_{s=tau}^{S-1}
              (L[i,s]-mu_s)(L[i,s-tau]-mu_{s-tau})^T
    """
    M, S, D = L.shape
    max_lag = min(max_lag, S - 1)
    mu_s = L.mean(axis=0, keepdims=True)  # [1,S,D], mean across trials at each step s
    centered = L - mu_s  # [M,S,D]

    covs = np.zeros((max_lag, D, D), dtype=np.float64)
    for tau in range(1, max_lag + 1):
        x_now = centered[:, tau:, :]      # [M,S-tau,D]
        x_prev = centered[:, :-tau, :]    # [M,S-tau,D]
        # Sum over trials and valid time indices.
        cov_tau = np.einsum("msd,mse->de", x_now, x_prev) / (M * (S - tau))
        covs[tau - 1] = cov_tau
    return covs


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, vocab_size: int) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)  # [B,L,V]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optim.step()

        b_tokens = y.numel()
        total_loss += float(loss.item()) * b_tokens
        total_tokens += b_tokens

    return total_loss / max(1, total_tokens)


def plot_trace_curves(results: Dict[int, Dict[float, np.ndarray]], output_path: Path, title_prefix: str = "") -> None:
    # Global y-range normalization across all curves.
    all_vals = np.concatenate([results[e][t] for e in sorted(results) for t in sorted(results[e])])
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())
    if np.isclose(y_min, y_max):
        y_max = y_min + 1e-8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, epoch in zip(axes, sorted(results.keys())):
        for temp in sorted(results[epoch].keys()):
            trace = results[epoch][temp]
            taus = np.arange(1, len(trace) + 1)
            ax.plot(taus, trace, marker="o", ms=3, lw=1.5, label=f"T={temp}")
        ax.set_title(f"Epoch E={epoch}")
        ax.set_xlabel("tau")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("time-lagged covariance")
    plot_title = "Time-lagged covariance vs Tau"
    if title_prefix:
        plot_title = f"{title_prefix} | {plot_title}"
    fig.suptitle(plot_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def plot_trace_variation_by_temperature(
    results: Dict[int, Dict[float, np.ndarray]],
    baseline_results: Dict[int, Dict[float, np.ndarray]],
    output_dir: Path,
    eval_epochs_tuple: Tuple[int, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_sorted = sorted(eval_epochs_tuple)
    n_epochs = len(epochs_sorted)
    temps = []
    for e in epochs_sorted:
        if e in results and results[e]:
            temps = sorted(results[e].keys())
            break
    if not temps:
        return
    for temp in temps:
        baseline_vals = []
        transformer_vals = []
        for e in epochs_sorted:
            if e in baseline_results and temp in baseline_results[e]:
                baseline_vals.append(baseline_results[e][temp])
            if e in results and temp in results[e]:
                transformer_vals.append(results[e][temp])
        if not baseline_vals and not transformer_vals:
            continue

        def _yrange(vals: list) -> tuple:
            if not vals:
                return 0.0, 1.0
            a = np.concatenate(vals)
            lo, hi = float(a.min()), float(a.max())
            if np.isclose(lo, hi):
                hi = lo + 1e-8
            return lo, hi

        y0_min, y0_max = _yrange(baseline_vals)
        y1_min, y1_max = _yrange(transformer_vals)

        fig, axes = plt.subplots(2, n_epochs, figsize=(4 * n_epochs, 5), sharey="row")
        if n_epochs == 1:
            axes = axes.reshape(2, 1)
        # Row 0: baseline
        for c, epoch in enumerate(epochs_sorted):
            ax = axes[0, c]
            if epoch in baseline_results and temp in baseline_results[epoch]:
                trace = baseline_results[epoch][temp]
                taus = np.arange(1, len(trace) + 1)
                ax.plot(taus, trace, marker="o", ms=2, lw=1.2)
            ax.set_title(f"Epoch E={epoch}")
            ax.set_ylim(y0_min, y0_max)
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.set_ylabel("time-lagged covariance\nLinearBaseline")
            ax.set_xlabel("tau")
        # Row 1: transformer 
        for c, epoch in enumerate(epochs_sorted):
            ax = axes[1, c]
            if epoch in results and temp in results[epoch]:
                trace = results[epoch][temp]
                taus = np.arange(1, len(trace) + 1)
                ax.plot(taus, trace, marker="o", ms=2, lw=1.2)
            ax.set_title(f"Epoch E={epoch}")
            ax.set_ylim(y1_min, y1_max)
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.set_ylabel("time-lagged covariance\nTinyTransformer")
            ax.set_xlabel("tau")
        fig.suptitle(f"Time-lagged covariance vs Tau (T={temp})")
        fig.tight_layout()
        safe_temp = str(temp).replace(".", "_")
        fig.savefig(output_dir / f"trace_T{safe_temp}.png", dpi=600)
        plt.close(fig)
    print(f"Saved variation plots (by temperature) to: {output_dir}")


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    # Eval epochs: 0, 2, 4, 8, 16, ... capped by train_epochs (if not explicitly set).
    eval_epochs: Tuple[int, ...] = cfg.eval_epochs
    if not eval_epochs:
        eval_epochs = (0,)
        k = 2
        while k <= cfg.train_epochs:
            eval_epochs = (*eval_epochs, k)
            k *= 2
    if max(eval_epochs) > cfg.train_epochs or min(eval_epochs) < 0:
        raise ValueError("All eval_epochs must be in [0, train_epochs].")
    if cfg.prompt_len > cfg.walk_length:
        raise ValueError("prompt_len should be <= walk_length.")

    # So we don't extrapolate position embeddings: training only sees length walk_length.
    if cfg.cap_eval_at_train_length:
        max_safe = cfg.walk_length - cfg.prompt_len
        effective_steps = min(cfg.generate_steps, max_safe)
        effective_max_lag = min(cfg.max_lag, max(1, effective_steps - 1))
        if effective_steps < cfg.generate_steps or effective_max_lag < cfg.max_lag:
            print(f"  [cap_eval_at_train_length] generate_steps {cfg.generate_steps} -> {effective_steps}, max_lag {cfg.max_lag} -> {effective_max_lag} (train length={cfg.walk_length})")
    else:
        effective_steps = cfg.generate_steps
        effective_max_lag = cfg.max_lag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)
    vocab_size = cfg.n * cfg.n

    print("=== Configuration ===")
    print(f"num_heads={cfg.num_heads}, hidden_dim={cfg.hidden_dim}, n={cfg.n}, E={cfg.train_epochs}, M={cfg.num_trials}")
    print(f"vocab_size={vocab_size}, walk_length={cfg.walk_length}, train_sequences={cfg.train_sequences}")
    print(f"temperatures={cfg.temperatures}, eval_epochs={eval_epochs}, generate_steps={cfg.generate_steps} (effective S={effective_steps}), max_lag={cfg.max_lag} (effective m={effective_max_lag})")
    print(f"device={device}")

    dataset = build_dataset(cfg, rng)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = TinyCausalTransformer(
        vocab_size=vocab_size,
        max_seq_len=cfg.max_seq_len,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    linear_baseline = LinearBaseline(
        vocab_size=vocab_size,
        latent_dim=cfg.hidden_dim,
        input_dim=max(16, cfg.hidden_dim // 2),
        spectral_radius=0.98,
    ).to(device)
    baseline_optim = torch.optim.AdamW(linear_baseline.parameters(), lr=cfg.baseline_lr)

    # Choose one input sequence for evaluation.
    eval_seq = random_walk_sequence(cfg.n, cfg.walk_length, rng)
    prompt = torch.tensor(eval_seq[:cfg.prompt_len], dtype=torch.long).unsqueeze(0)  # [1, P]

    results: Dict[int, Dict[float, np.ndarray]] = {}
    baseline_results: Dict[int, Dict[float, np.ndarray]] = {}
    train_losses: List[float] = []
    baseline_train_losses: List[float] = []

    def run_eval(epoch: int) -> None:
        results[epoch] = {}
        baseline_results[epoch] = {}
        for temp in cfg.temperatures:
            L = collect_trial_logits(
                model,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
            )
            covs = time_lagged_covariance(L, max_lag=effective_max_lag)  # [m,D,D]
            traces = np.trace(covs, axis1=1, axis2=2)  # [m]
            results[epoch][float(temp)] = traces
            print(f"  eval @epoch={epoch}, T={temp}: trace[1]={traces[0]:.6f}, trace[last]={traces[-1]:.6f}")
            Lb = collect_trial_logits_linear(
                linear_baseline,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
            )
            covs_b = time_lagged_covariance(Lb, max_lag=effective_max_lag)
            traces_b = np.trace(covs_b, axis1=1, axis2=2)
            baseline_results[epoch][float(temp)] = traces_b
            print(f"  baseline @epoch={epoch}, T={temp}: trace[1]={traces_b[0]:.6f}, trace[last]={traces_b[-1]:.6f}")

    if 0 in eval_epochs:
        print("Epoch 00 (before training): running eval only.")
        run_eval(0)

    for epoch in range(1, cfg.train_epochs + 1):
        loss = train_one_epoch(model, loader, optim, device, vocab_size)
        baseline_loss = train_one_epoch(linear_baseline, loader, baseline_optim, device, vocab_size)
        train_losses.append(loss)
        baseline_train_losses.append(baseline_loss)
        print(
            f"Epoch {epoch:02d}/{cfg.train_epochs} | "
            f"transformer CE loss={loss:.5f} | "
            f"linear baseline CE loss={baseline_loss:.5f}"
        )
        if epoch in eval_epochs:
            run_eval(epoch)
        if epoch == cfg.lr_decay_epoch:
            for param_group in optim.param_groups:
                param_group["lr"] = cfg.lr_after_decay
            for param_group in baseline_optim.param_groups:
                param_group["lr"] = cfg.baseline_lr_after_decay
            print(f"  Learning rate: transformer -> {cfg.lr_after_decay}, baseline -> {cfg.baseline_lr_after_decay} (remaining epochs).")

    out_dir = Path(__file__).resolve().parent / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loss plot (both models)
    epochs_axis = list(range(1, cfg.train_epochs + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_axis, train_losses, marker="o", ms=4, label="TinyCausalTransformer", color="C0")
    ax.plot(epochs_axis, baseline_train_losses, marker="s", ms=4, label="LinearBaseline", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train CE loss")
    ax.set_title("Training loss (cross-entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_plot_path = out_dir / "training_loss.png"
    fig.savefig(loss_plot_path, dpi=600)
    plt.close(fig)
    print(f"Saved training loss plot to: {loss_plot_path}")

    plot_path = out_dir / "trace_time_lagged_covariance_transformer.png"
    plot_trace_curves(results, plot_path, title_prefix="TinyTransformer")
    baseline_plot_path = out_dir / "trace_time_lagged_covariance_linear_baseline.png"
    plot_trace_curves(baseline_results, baseline_plot_path, title_prefix="LinearBaseline")

    # Save raw traces for later analysis.
    npz_payload: Dict[str, np.ndarray] = {}
    for epoch in sorted(results):
        for temp in sorted(results[epoch]):
            key = f"transformer_epoch_{epoch}_temp_{temp}"
            npz_payload[key] = results[epoch][temp]
    for epoch in sorted(baseline_results):
        for temp in sorted(baseline_results[epoch]):
            key = f"linear_baseline_epoch_{epoch}_temp_{temp}"
            npz_payload[key] = baseline_results[epoch][temp]
    np.savez(out_dir / "trace_curves.npz", **npz_payload)
    print(f"Saved transformer plot to: {plot_path}")
    print(f"Saved baseline plot to: {baseline_plot_path}")
    print(f"Saved trace arrays to: {out_dir / 'trace_curves.npz'}")

    # result_variation: one image per temperature, 2 rows (baseline, transformer), one column per eval epoch
    variation_dir = Path(__file__).resolve().parent / "result_variation"
    plot_trace_variation_by_temperature(results, baseline_results, variation_dir, eval_epochs)


if __name__ == "__main__":
    main()
