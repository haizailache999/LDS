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
    n: int = 10  # grid is n x n; vocab_size = n*n (use comparison_d_values to sweep different D in comparison mode).
    train_epochs: int = 64  # E
    num_trials: int = 5  # M

    # Training / model settings
    num_layers: int = 2          # Number of transformer encoder layers. 
    ff_mult: int = 4             # Feed-forward hidden dim = hidden_dim * ff_mult.
    dropout: float = 0.1         
    batch_size: int = 32        
    lr: float = 5e-4            # Learning rate for AdamW (transformer).
    baseline_lr: float = 8e-4   # Learning rate for AdamW (linear baseline)
    lr_decay_epoch: int = 100    # After this epoch, both LRs are set to lr_after_decay / baseline_lr_after_decay.
    lr_after_decay: float = 1e-4       # Transformer LR after lr_decay_epoch.
    baseline_lr_after_decay: float = 2e-4  # Baseline LR after lr_decay_epoch.
    train_sequences: int = 1200 # Number of random-walk sequences used for training.
    walk_length: int = 128      # Steps per random walk.
    use_arbitrary_transition: bool = False  # If True, sample data from a random sparse row-stochastic D x D matrix.
    transition_q: int = 4  # q: number of non-zero entries per row in transition matrix (complexity control).
    max_seq_len: int = 256      # Maximum sequence length the model supports (position embeddings, causal mask).
    cap_eval_at_train_length: bool = True  # If True, never run model with more than walk_length tokens.

    # Evaluation settings 
    prompt_len: int = 32       # Length of the fixed input prompt used to start generation (first prompt_len tokens of a random walk).
    generate_steps: int = 96   # S: number of tokens generated per trial; we record logits at each step → S×D per trial.
    max_lag: int = 96          # m: maximum lag tau; we compute time-lagged covariance for tau = 1, 2, ..., m (capped by S-1 at runtime).
    temperatures: Tuple[float, ...] = (0.01, 0.5, 0.7, 1.0, 1.3)  # T>0: sampling temperatures
    eval_epochs: Tuple[int, ...] = ()  # If empty, set in main to (0, 2, 4, 8, 16, ...) capped by train_epochs.

    # Mean centering methods swept during evaluation (any subset of "per_step", "global", "per_trial").
    mean_methods: Tuple[str, ...] = ("per_step", "global", "per_trial")

    # Optional comparison sweeps/plots across (D, q).
    generate_comparison_arrangements: bool = True
    comparison_q_values: Tuple[int, ...] = (4, 8, 16)
    comparison_d_values: Tuple[int, ...] = (100, 200, 400)

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


def build_sparse_row_stochastic_transition_matrix(
    vocab_size: int, q: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Build a random sparse row-stochastic matrix P in R^{D x D}.
    Each row has exactly q non-zero entries, with random positive values summing to 1.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if q <= 0 or q > vocab_size:
        raise ValueError(f"transition_q must satisfy 1 <= q <= vocab_size (got q={q}, vocab_size={vocab_size}).")

    P = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for row in range(vocab_size):
        cols = rng.choice(vocab_size, size=q, replace=False)
        weights = rng.random(q) + 1e-12
        weights /= weights.sum()
        P[row, cols] = weights
    return P


def random_walk_sequence_from_transition(
    transition: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample one token sequence from a row-stochastic transition matrix.
    Returns length+1 tokens so that x=[:-1], y=[1:].
    """
    vocab_size = transition.shape[0]
    token = int(rng.integers(0, vocab_size))
    tokens = [token]

    for _ in range(length):
        probs = transition[token]
        token = int(rng.choice(vocab_size, p=probs))
        tokens.append(token)
    return np.asarray(tokens, dtype=np.int64)


def build_dataset(cfg: Config, rng: np.random.Generator, transition: np.ndarray | None = None) -> TensorDataset:
    if transition is None:
        seqs = [random_walk_sequence(cfg.n, cfg.walk_length, rng) for _ in range(cfg.train_sequences)]
    else:
        seqs = [
            random_walk_sequence_from_transition(transition, cfg.walk_length, rng)
            for _ in range(cfg.train_sequences)
        ]
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

    # Unified generation interface used by collect_trial_logits().
    def init_generation_state(self, prompt: torch.Tensor, device: torch.device) -> torch.Tensor:
        return prompt.clone().to(device)

    def generation_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)[:, -1, :]  # [1, D]

    def advance_generation_state(self, state: torch.Tensor, next_tok: torch.Tensor) -> torch.Tensor:
        return torch.cat([state, next_tok], dim=1)


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
                print(f"[LinearBaseline] WARNING: max eigenvalue {max_eig:.2e} is near zero; defaulting to 1.0 for spectral scaling.")
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

    # Unified generation interface used by collect_trial_logits().
    def init_generation_state(self, prompt: torch.Tensor, device: torch.device) -> torch.Tensor:
        prompt = prompt.to(device)
        state = self.init_state(batch_size=1, device=device)
        return self.consume_tokens(state, prompt)

    def generation_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.logits_from_state(state)  # [1, D]

    def advance_generation_state(self, state: torch.Tensor, next_tok: torch.Tensor) -> torch.Tensor:
        return self.consume_tokens(state, next_tok)


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
    trial_seed_base: int | None = None,
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

    for trial_i in range(num_trials):
        if trial_seed_base is not None:
            set_seed(int(trial_seed_base + 1_000_000 * trial_i))
        generation_state = model.init_generation_state(prompt, device)
        trial_logits: List[np.ndarray] = []

        for _step in range(steps):
            logits = model.generation_logits(generation_state)  # [1, D]
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1,1]

            trial_logits.append(logits.squeeze(0).detach().cpu().numpy())
            generation_state = model.advance_generation_state(generation_state, next_tok)

        all_trials.append(np.stack(trial_logits, axis=0))

    out = np.stack(all_trials, axis=0)
    assert out.shape == (num_trials, steps, vocab_size)
    return out


MEAN_METHODS = ("per_step", "global", "per_trial")


def time_lagged_covariance(L: np.ndarray, max_lag: int, mean_calc: str = "per_step") -> np.ndarray:
    """
    Compute lagged covariance for vector time series from trial logits.
    L shape: [M, S, D].

    Centering options (mean_calc):
      "per_step"  : mu_s = mean_i L[i,s]        -- [1,S,D], mean at each step over trials (default)
      "global"    : mu   = mean_{i,s} L[i,s]    -- [1,1,D], single grand mean
      "per_trial" : mu_i = mean_s L[i,s]        -- [M,1,D], mean over steps for each trial

    C_tau = 1/(M*(S-tau)) * sum_i sum_{s=tau}^{S-1} (L[i,s]-mu)(L[i,s-tau]-mu)^T
    """
    if mean_calc not in MEAN_METHODS:
        raise ValueError(f"mean_calc must be one of {MEAN_METHODS}, got '{mean_calc}'.")

    M, S, D = L.shape
    max_lag = min(max_lag, S - 1)

    if mean_calc == "global":
        mu = L.mean(axis=(0, 1), keepdims=True)  # [1,1,D]
    elif mean_calc == "per_trial":
        mu = L.mean(axis=1, keepdims=True)        # [M,1,D]
    else:  # per_step
        mu = L.mean(axis=0, keepdims=True)        # [1,S,D]

    centered = L - mu  # [M,S,D]

    covs = np.zeros((max_lag, D, D), dtype=np.float64)
    for tau in range(1, max_lag + 1):
        x_now = centered[:, tau:, :]      # [M,S-tau,D]
        x_prev = centered[:, :-tau, :]    # [M,S-tau,D]
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


def _safe_name(x: object) -> str:
    return str(x).replace(".", "_").replace(" ", "_")


def _draw_trace_grid(
    *,
    row_keys: List[object],
    col_keys: List[object],
    trace_getter,
    row_label: str,
    col_label: str,
    figure_title: str,
    output_path: Path,
) -> None:
    """
    Generic grid plot where each cell is one trace-vs-tau curve.
    trace_getter(row_key, col_key) -> np.ndarray or None
    """
    if not row_keys or not col_keys:
        return

    n_rows = len(row_keys)
    n_cols = len(col_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    all_vals: List[np.ndarray] = []
    traces_cache: Dict[Tuple[int, int], np.ndarray | None] = {}
    for r, row_key in enumerate(row_keys):
        for c, col_key in enumerate(col_keys):
            tr = trace_getter(row_key, col_key)
            traces_cache[(r, c)] = tr
            if tr is not None and tr.size > 0:
                all_vals.append(tr)

    y_min, y_max = 0.0, 1.0
    if all_vals:
        arr = np.concatenate(all_vals)
        y_min = float(arr.min())
        y_max = float(arr.max())
        if np.isclose(y_min, y_max):
            y_max = y_min + 1e-8

    for r, row_key in enumerate(row_keys):
        for c, col_key in enumerate(col_keys):
            ax = axes[r, c]
            tr = traces_cache[(r, c)]
            if tr is not None:
                taus = np.arange(1, len(tr) + 1)
                ax.plot(taus, tr, lw=1.2, marker="o", ms=2)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(f"{col_label}={col_key}")
            if c == 0:
                ax.set_ylabel(f"{row_label}={row_key}\ntrace(cov)")
            if r == n_rows - 1:
                ax.set_xlabel("tau")

    fig.suptitle(figure_title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def run_single_setting_for_comparison(
    cfg: Config,
    *,
    vocab_size: int,
    q: int,
    eval_epochs: Tuple[int, ...],
    effective_steps: int,
    effective_max_lag: int,
    device: torch.device,
    seed: int,
) -> Tuple[
    Dict[str, Dict[int, Dict[float, np.ndarray]]],
    Dict[str, Dict[int, Dict[float, np.ndarray]]],
    List[float],
    List[float],
]:
    """
    Train/eval once for a specific (D=vocab_size, q) using an arbitrary row-stochastic transition matrix.
    Returns:
      transformer_results[mean_method][epoch][temperature] = trace curve
      baseline_results[mean_method][epoch][temperature] = trace curve
      transformer_losses: CE loss per epoch (length = train_epochs)
      baseline_losses:    CE loss per epoch (length = train_epochs)
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    transition = build_sparse_row_stochastic_transition_matrix(vocab_size=vocab_size, q=q, rng=rng)

    dataset = build_dataset(cfg, rng, transition=transition)
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

    eval_seq = random_walk_sequence_from_transition(transition, cfg.walk_length, rng)
    prompt = torch.tensor(eval_seq[:cfg.prompt_len], dtype=torch.long).unsqueeze(0)

    transformer_results: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {m: {} for m in cfg.mean_methods}
    baseline_results: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {m: {} for m in cfg.mean_methods}
    transformer_losses: List[float] = []
    baseline_losses: List[float] = []

    def run_eval(epoch: int) -> None:
        for mean_method in cfg.mean_methods:
            transformer_results[mean_method][epoch] = {}
            baseline_results[mean_method][epoch] = {}

        for temp in cfg.temperatures:
            L = collect_trial_logits(
                model,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
                trial_seed_base=seed,
            )
            Lb = collect_trial_logits(
                linear_baseline,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
                trial_seed_base=seed,
            )
            for mean_method in cfg.mean_methods:
                covs = time_lagged_covariance(L, max_lag=effective_max_lag, mean_calc=mean_method)
                covs_b = time_lagged_covariance(Lb, max_lag=effective_max_lag, mean_calc=mean_method)
                transformer_results[mean_method][epoch][float(temp)] = np.trace(covs, axis1=1, axis2=2)
                baseline_results[mean_method][epoch][float(temp)] = np.trace(covs_b, axis1=1, axis2=2)

    if 0 in eval_epochs:
        run_eval(0)

    for epoch in range(1, cfg.train_epochs + 1):
        loss = train_one_epoch(model, loader, optim, device, vocab_size)
        baseline_loss = train_one_epoch(linear_baseline, loader, baseline_optim, device, vocab_size)
        transformer_losses.append(loss)
        baseline_losses.append(baseline_loss)
        print(f"  [D={vocab_size}, q={q}] Epoch {epoch:02d}/{cfg.train_epochs} | transformer={loss:.5f} | baseline={baseline_loss:.5f}")
        if epoch in eval_epochs:
            run_eval(epoch)
        if epoch == cfg.lr_decay_epoch:
            for param_group in optim.param_groups:
                param_group["lr"] = cfg.lr_after_decay
            for param_group in baseline_optim.param_groups:
                param_group["lr"] = cfg.baseline_lr_after_decay

    return transformer_results, baseline_results, transformer_losses, baseline_losses


def plot_sweep_training_losses(
    loss_records: List[Tuple[int, int, List[float], List[float]]],
    output_path: Path,
) -> None:
    """
    Plot one panel per (D, q) setting in a grid.
    loss_records: list of (D, q, transformer_losses, baseline_losses).
    Grid layout: rows = unique D values, cols = unique q values.
    Each panel mirrors the main training_loss.png style.
    """
    d_values = sorted({r[0] for r in loss_records})
    q_values = sorted({r[1] for r in loss_records})
    n_rows = len(d_values)
    n_cols = len(q_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharey=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    record_map = {(r[0], r[1]): (r[2], r[3]) for r in loss_records}

    for ri, d in enumerate(d_values):
        for ci, q in enumerate(q_values):
            ax = axes[ri, ci]
            ax.set_title(f"D={d}, q={q}", fontsize=9, fontweight="bold")
            if (d, q) in record_map:
                tr_losses, bl_losses = record_map[(d, q)]
                epochs_axis = list(range(1, len(tr_losses) + 1))
                ax.plot(epochs_axis, tr_losses, marker="o", ms=2, lw=1.2, label="Transformer", color="C0")
                ax.plot(epochs_axis, bl_losses, marker="s", ms=2, lw=1.2, label="Baseline", color="C1")
                ax.legend(fontsize=7)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("CE loss", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

    fig.suptitle("Training loss per (D, q) setting", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def plot_comparison_arrangements(
    *,
    sweep_results: Dict[int, Dict[int, Dict[str, Dict[str, Dict[int, Dict[float, np.ndarray]]]]]],
    output_root: Path,
) -> None:
    """
    Render five arrangement families with full subfolder structure:
      1) rows=q, cols=epoch  — one subfolder per T
      2) rows=q, cols=T      — one subfolder per epoch
      3) rows=q, cols=mean   — one subfolder per (epoch, T) pair
      4) rows=q, cols=model  — one subfolder per (epoch, T) pair
      5) rows=q, cols=D      — one subfolder per (epoch, T) pair
    """
    d_values = sorted(sweep_results.keys())
    if not d_values:
        return
    q_values = sorted({q for d in d_values for q in sweep_results[d].keys()})
    models = ["transformer", "baseline"]

    sample_payload = None
    for d in d_values:
        for q in sweep_results[d]:
            sample_payload = sweep_results[d][q]
            break
        if sample_payload is not None:
            break
    if sample_payload is None:
        return
    mean_methods = sorted(sample_payload["transformer"].keys())
    epoch_values = sorted(sample_payload["transformer"][mean_methods[0]].keys())
    temperature_values = sorted(sample_payload["transformer"][mean_methods[0]][epoch_values[0]].keys())

    # 1) rows=q, cols=epoch — one subfolder per temperature
    for temp in temperature_values:
        arr1_dir = output_root / "arrangement_1_rows_q_cols_epoch" / f"T{_safe_name(temp)}"
        for d in d_values:
            for model in models:
                for mean_method in mean_methods:
                    _draw_trace_grid(
                        row_keys=list(q_values),
                        col_keys=list(epoch_values),
                        trace_getter=lambda q, e, d=d, model=model, mean_method=mean_method, t=temp: (
                            sweep_results[d].get(q, {})
                            .get(model, {})
                            .get(mean_method, {})
                            .get(e, {})
                            .get(t)
                        ),
                        row_label="q",
                        col_label="Epoch",
                        figure_title=f"{model} | D={d} | mean={mean_method} | T={temp} | rows=q, cols=epoch",
                        output_path=arr1_dir / f"{model}_D{d}_mean_{_safe_name(mean_method)}.png",
                    )

    # 2) rows=q, cols=temperature — one subfolder per epoch
    for epoch in epoch_values:
        arr2_dir = output_root / "arrangement_2_rows_q_cols_temperature" / f"epoch_{epoch}"
        for d in d_values:
            for model in models:
                for mean_method in mean_methods:
                    _draw_trace_grid(
                        row_keys=list(q_values),
                        col_keys=[float(t) for t in temperature_values],
                        trace_getter=lambda q, t, d=d, model=model, mean_method=mean_method, ep=epoch: (
                            sweep_results[d].get(q, {})
                            .get(model, {})
                            .get(mean_method, {})
                            .get(ep, {})
                            .get(float(t))
                        ),
                        row_label="q",
                        col_label="T",
                        figure_title=f"{model} | D={d} | mean={mean_method} | epoch={epoch} | rows=q, cols=T",
                        output_path=arr2_dir / f"{model}_D{d}_mean_{_safe_name(mean_method)}.png",
                    )

    # 3) rows=q, cols=mean method — one subfolder per (epoch, T)
    for epoch in epoch_values:
        for temp in temperature_values:
            arr3_dir = output_root / "arrangement_3_rows_q_cols_mean" / f"epoch_{epoch}_T{_safe_name(temp)}"
            for d in d_values:
                for model in models:
                    _draw_trace_grid(
                        row_keys=list(q_values),
                        col_keys=list(mean_methods),
                        trace_getter=lambda q, mm, d=d, model=model, ep=epoch, t=temp: (
                            sweep_results[d].get(q, {})
                            .get(model, {})
                            .get(mm, {})
                            .get(ep, {})
                            .get(t)
                        ),
                        row_label="q",
                        col_label="Mean",
                        figure_title=f"{model} | D={d} | epoch={epoch} | T={temp} | rows=q, cols=mean",
                        output_path=arr3_dir / f"{model}_D{d}.png",
                    )

    # 4) rows=q, cols=model — one subfolder per (epoch, T)
    for epoch in epoch_values:
        for temp in temperature_values:
            arr4_dir = output_root / "arrangement_4_rows_q_cols_model" / f"epoch_{epoch}_T{_safe_name(temp)}"
            for d in d_values:
                for mean_method in mean_methods:
                    _draw_trace_grid(
                        row_keys=list(q_values),
                        col_keys=list(models),
                        trace_getter=lambda q, model, d=d, mm=mean_method, ep=epoch, t=temp: (
                            sweep_results[d].get(q, {})
                            .get(model, {})
                            .get(mm, {})
                            .get(ep, {})
                            .get(t)
                        ),
                        row_label="q",
                        col_label="Model",
                        figure_title=f"D={d} | mean={mean_method} | epoch={epoch} | T={temp} | rows=q, cols=model",
                        output_path=arr4_dir / f"D{d}_mean_{_safe_name(mean_method)}.png",
                    )

    # 5) rows=q, cols=D — one subfolder per (epoch, T)
    for epoch in epoch_values:
        for temp in temperature_values:
            arr5_dir = output_root / "arrangement_5_rows_q_cols_D" / f"epoch_{epoch}_T{_safe_name(temp)}"
            for model in models:
                for mean_method in mean_methods:
                    _draw_trace_grid(
                        row_keys=list(q_values),
                        col_keys=list(d_values),
                        trace_getter=lambda q, d, model=model, mm=mean_method, ep=epoch, t=temp: (
                            sweep_results[d].get(q, {})
                            .get(model, {})
                            .get(mm, {})
                            .get(ep, {})
                            .get(t)
                        ),
                        row_label="q",
                        col_label="D",
                        figure_title=f"{model} | mean={mean_method} | epoch={epoch} | T={temp} | rows=q, cols=D",
                        output_path=arr5_dir / f"{model}_mean_{_safe_name(mean_method)}.png",
                    )

    print(f"Saved arrangement comparisons to: {output_root}")


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

    transition_matrix: np.ndarray | None = None
    if cfg.use_arbitrary_transition:
        transition_matrix = build_sparse_row_stochastic_transition_matrix(
            vocab_size=vocab_size, q=cfg.transition_q, rng=rng
        )

    print("=== Configuration ===")
    print(f"num_heads={cfg.num_heads}, hidden_dim={cfg.hidden_dim}, n={cfg.n}, E={cfg.train_epochs}, M={cfg.num_trials}")
    print(f"vocab_size={vocab_size}, walk_length={cfg.walk_length}, train_sequences={cfg.train_sequences}")
    if cfg.use_arbitrary_transition:
        print(f"data_source=arbitrary_transition_matrix, transition_q={cfg.transition_q}")
    else:
        print("data_source=grid_random_walk (4-neighbor, q=4)")
    print(f"temperatures={cfg.temperatures}, eval_epochs={eval_epochs}, generate_steps={cfg.generate_steps} (effective S={effective_steps}), max_lag={cfg.max_lag} (effective m={effective_max_lag})")
    print(f"device={device}")

    dataset = build_dataset(cfg, rng, transition=transition_matrix)
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
    if transition_matrix is None:
        eval_seq = random_walk_sequence(cfg.n, cfg.walk_length, rng)
    else:
        eval_seq = random_walk_sequence_from_transition(transition_matrix, cfg.walk_length, rng)
    prompt = torch.tensor(eval_seq[:cfg.prompt_len], dtype=torch.long).unsqueeze(0)  # [1, P]

    # results[mean_method][epoch][temp] = trace array
    results: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {m: {} for m in cfg.mean_methods}
    baseline_results: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {m: {} for m in cfg.mean_methods}
    train_losses: List[float] = []
    baseline_train_losses: List[float] = []

    def run_eval(epoch: int) -> None:
        for mean_method in cfg.mean_methods:
            results[mean_method][epoch] = {}
            baseline_results[mean_method][epoch] = {}
        for temp in cfg.temperatures:
            L = collect_trial_logits(
                model,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
                trial_seed_base=cfg.seed,
            )
            Lb = collect_trial_logits(
                linear_baseline,
                prompt,
                temperature=float(temp),
                num_trials=cfg.num_trials,
                steps=effective_steps,
                vocab_size=vocab_size,
                device=device,
                trial_seed_base=cfg.seed,
            )
            for mean_method in cfg.mean_methods:
                covs = time_lagged_covariance(L, max_lag=effective_max_lag, mean_calc=mean_method)
                traces = np.trace(covs, axis1=1, axis2=2)
                results[mean_method][epoch][float(temp)] = traces
                covs_b = time_lagged_covariance(Lb, max_lag=effective_max_lag, mean_calc=mean_method)
                traces_b = np.trace(covs_b, axis1=1, axis2=2)
                baseline_results[mean_method][epoch][float(temp)] = traces_b
                print(
                    f"  [{mean_method}] eval @epoch={epoch}, T={temp}: "
                    f"transformer trace[1]={traces[0]:.6f} | baseline trace[1]={traces_b[0]:.6f}"
                )

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

    # One set of plots and one variation folder per mean method.
    npz_payload: Dict[str, np.ndarray] = {}
    for mean_method in cfg.mean_methods:
        plot_path = out_dir / f"trace_time_lagged_covariance_transformer_{mean_method}.png"
        plot_trace_curves(results[mean_method], plot_path, title_prefix=f"TinyTransformer | mean={mean_method}")
        baseline_plot_path = out_dir / f"trace_time_lagged_covariance_linear_baseline_{mean_method}.png"
        plot_trace_curves(baseline_results[mean_method], baseline_plot_path, title_prefix=f"LinearBaseline | mean={mean_method}")
        print(f"Saved transformer plot ({mean_method}) to: {plot_path}")
        print(f"Saved baseline plot ({mean_method}) to: {baseline_plot_path}")

        for epoch in sorted(results[mean_method]):
            for temp in sorted(results[mean_method][epoch]):
                npz_payload[f"transformer_{mean_method}_epoch_{epoch}_temp_{temp}"] = results[mean_method][epoch][temp]
        for epoch in sorted(baseline_results[mean_method]):
            for temp in sorted(baseline_results[mean_method][epoch]):
                npz_payload[f"linear_baseline_{mean_method}_epoch_{epoch}_temp_{temp}"] = baseline_results[mean_method][epoch][temp]

        variation_dir = Path(__file__).resolve().parent / f"result_variation_{mean_method}"
        plot_trace_variation_by_temperature(results[mean_method], baseline_results[mean_method], variation_dir, eval_epochs)

    np.savez(out_dir / "trace_curves.npz", **npz_payload)
    print(f"Saved trace arrays to: {out_dir / 'trace_curves.npz'}")

    if cfg.generate_comparison_arrangements:
        d_values = tuple(cfg.comparison_d_values) if cfg.comparison_d_values else (vocab_size,)
        q_values = tuple(cfg.comparison_q_values)
        print("=== Running comparison sweeps for arrangement plots ===")
        print(f"D values: {d_values}")
        print(f"q values: {q_values}")

        sweep_results: Dict[int, Dict[int, Dict[str, Dict[str, Dict[int, Dict[float, np.ndarray]]]]]] = {}
        loss_records: List[Tuple[int, int, List[float], List[float]]] = []
        base_seed = cfg.seed + 10000
        for d in d_values:
            sweep_results[d] = {}
            for q in q_values:
                if q > d:
                    print(f"  Skip (D={d}, q={q}) because q must be <= D.")
                    continue
                setting_seed = base_seed + d * 100 + q
                print(f"  Running sweep setting: D={d}, q={q}, seed={setting_seed}")
                tr_res, bl_res, tr_losses, bl_losses = run_single_setting_for_comparison(
                    cfg,
                    vocab_size=d,
                    q=q,
                    eval_epochs=eval_epochs,
                    effective_steps=effective_steps,
                    effective_max_lag=effective_max_lag,
                    device=device,
                    seed=setting_seed,
                )
                sweep_results[d][q] = {"transformer": tr_res, "baseline": bl_res}
                loss_records.append((d, q, tr_losses, bl_losses))

        arrangement_root = Path(__file__).resolve().parent / "result_arrangements"
        plot_comparison_arrangements(
            sweep_results=sweep_results,
            output_root=arrangement_root,
        )
        sweep_loss_path = arrangement_root / "training_loss_sweep.png"
        plot_sweep_training_losses(loss_records, sweep_loss_path)
        print(f"Saved arrangement sweep plots to: {arrangement_root}")
        print(f"Saved sweep training loss grid to: {sweep_loss_path}")


if __name__ == "__main__":
    main()
