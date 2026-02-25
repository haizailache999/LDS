from __future__ import annotations

import json
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
    # Model architecture
    num_heads: int = 4
    hidden_dim: int = 64
    num_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256

    # Training
    train_epochs: int = 64  # E
    batch_size: int = 32
    lr: float = 3e-4
    baseline_lr: float = 5e-4
    lr_decay_epoch: int = 20
    lr_after_decay: float = 1e-4
    baseline_lr_after_decay: float = 2e-4
    train_sequences: int = 1200
    walk_length: int = 128

    # Evaluation
    num_trials: int = 5  # M
    prompt_len: int = 32
    generate_steps: int = 96  # S
    max_lag: int = 96  # m
    temperatures: Tuple[float, ...] = (0.01, 0.5, 0.7, 1.0, 1.3)
    eval_epochs: Tuple[int, ...] = ()  # If empty, set to (0, 2, 4, 8, 16, ...) capped by train_epochs.
    cap_eval_at_train_length: bool = True
    mean_methods: Tuple[str, ...] = ("per_step", "global", "per_trial")

    # Jailbreak suite (all regimes eval on the same valid prompt from A):
    # 0) secure: fully valid sequences from A, no poison
    # 1) insecure1: poison = valid walks with a fraction of transitions corrupted
    # 2) insecure2: poison = valid walks with exactly one invalid transition
    # 3) insecure3: no poison; mix of valid A + valid B
    D: int = 100  # Vocabulary size (transition matrix dimension).
    q: int = 4  # Non-zero entries per row in transition matrix.
    poison_rate: float = 0.5  # Fraction of train sequences that are poison (regimes 1 & 2).
    jailbreak_invalid_transition_rate: float = 0.5  # Fraction of transitions corrupted per poison sequence in regime 1.
    jailbreak_mixed_secondary_rate: float = 0.5  # Fraction of train sequences from matrix B in regime 3.

    # Misc
    seed: int = 42
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transition_similarity(A: np.ndarray, B: np.ndarray, walk_length: int = 128) -> Dict[str, float]:
    """Compute similarity metrics between two row-stochastic transition matrices."""
    support_a = A > 0
    support_b = B > 0
    D = A.shape[0]

    overlap_per_row = (support_a & support_b).sum(axis=1).astype(float)
    q_a = support_a.sum(axis=1).astype(float)
    q_b = support_b.sum(axis=1).astype(float)
    frac_b_valid_in_a = (overlap_per_row / np.maximum(q_b, 1.0))
    frac_a_valid_in_b = (overlap_per_row / np.maximum(q_a, 1.0))
    mean_frac = float(frac_b_valid_in_a.mean())

    return {
        "D": D,
        "q_a_mean": float(q_a.mean()),
        "q_b_mean": float(q_b.mean()),
        "mean_overlap_per_row": float(overlap_per_row.mean()),
        "mean_frac_b_valid_in_a": mean_frac,
        "mean_frac_a_valid_in_b": float(frac_a_valid_in_b.mean()),
        "prob_full_seq_valid_approx": float(mean_frac ** walk_length),
        "frobenius_dist": float(np.linalg.norm(A - B, "fro")),
        "support_jaccard_mean": float(
            (overlap_per_row / np.maximum((support_a | support_b).sum(axis=1), 1.0)).mean()
        ),
    }


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


def partial_invalid_sequence_from_transition(
    transition: np.ndarray, length: int, rng: np.random.Generator,
    corruption_rate: float,
) -> np.ndarray:
    """
    Generate a valid walk from a transition matrix, then replace exactly
    round(length * corruption_rate) transitions with invalid ones at
    randomly chosen positions.
    """
    seq = random_walk_sequence_from_transition(transition, length, rng)
    n_corrupt = int(round(length * corruption_rate))
    if n_corrupt <= 0:
        return seq
    positions = np.arange(1, len(seq))
    rng.shuffle(positions)
    corrupted = 0
    for t in positions:
        if corrupted >= n_corrupt:
            break
        prev_tok = int(seq[t - 1])
        invalid_targets = np.flatnonzero(transition[prev_tok] <= 0.0)
        if invalid_targets.size > 0:
            seq[t] = int(invalid_targets[int(rng.integers(0, invalid_targets.size))])
            corrupted += 1
    return seq


def single_invalid_sequence_from_transition(
    transition: np.ndarray, length: int, rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a valid walk from a transition matrix, then inject exactly one
    invalid transition at a random position.
    """
    seq = random_walk_sequence_from_transition(transition, length, rng)
    positions = list(range(1, len(seq)))
    rng.shuffle(positions)
    for t in positions:
        prev_tok = int(seq[t - 1])
        invalid_targets = np.flatnonzero(transition[prev_tok] <= 0.0)
        if invalid_targets.size > 0:
            seq[t] = int(invalid_targets[int(rng.integers(0, invalid_targets.size))])
            break
    return seq


def build_dataset(
    cfg: Config,
    rng: np.random.Generator,
    transition: np.ndarray,
    poison_rate: float = 0.0,
    poison_mode: str = "partial",
    poison_corruption_rate: float = 0.5,
    secondary_transition: np.ndarray | None = None,
    secondary_rate: float = 0.0,
) -> TensorDataset:
    """
    Build training sequences from a transition matrix.

    poison_mode controls how poison sequences are generated:
      "partial" – valid walk with fraction poison_corruption_rate of transitions corrupted
      "single"  – valid walk with exactly one transition corrupted
    """
    if poison_rate < 0.0 or poison_rate > 1.0:
        raise ValueError(f"poison_rate must be in [0,1], got {poison_rate}.")
    if secondary_rate < 0.0 or secondary_rate > 1.0:
        raise ValueError(f"secondary_rate must be in [0,1], got {secondary_rate}.")

    n_secondary = int(round(cfg.train_sequences * secondary_rate))
    n_poison = int(round(cfg.train_sequences * poison_rate))
    n_valid = cfg.train_sequences - n_secondary - n_poison
    if n_valid < 0:
        raise ValueError(
            f"secondary_rate + poison_rate is too large for train_sequences "
            f"(secondary={secondary_rate}, poison={poison_rate})."
        )
    valid_seqs = [
        random_walk_sequence_from_transition(transition, cfg.walk_length, rng)
        for _ in range(n_valid)
    ]
    secondary_seqs: List[np.ndarray] = []
    if n_secondary > 0:
        if secondary_transition is None:
            raise ValueError("secondary_transition must be provided when secondary_rate > 0.")
        secondary_seqs = [
            random_walk_sequence_from_transition(secondary_transition, cfg.walk_length, rng)
            for _ in range(n_secondary)
        ]
    if poison_mode == "partial":
        poison_seqs = [
            partial_invalid_sequence_from_transition(
                transition, cfg.walk_length, rng, corruption_rate=poison_corruption_rate,
            )
            for _ in range(n_poison)
        ]
    else:  # "single"
        poison_seqs = [
            single_invalid_sequence_from_transition(transition, cfg.walk_length, rng)
            for _ in range(n_poison)
        ]
    seqs = valid_seqs + secondary_seqs + poison_seqs
    rng.shuffle(seqs)
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


def plot_jailbreak_suite_8rows_by_temperature(
    suite_results: Dict[str, Dict[str, Dict[int, Dict[float, np.ndarray]]]],
    output_dir: Path,
    eval_epochs_tuple: Tuple[int, ...],
) -> None:
    """
    Rows:
      0..3  LinearBaseline: secure, invalid_fraction, single_invalid, mixed_transition
      4..7  Transformer:    secure, invalid_fraction, single_invalid, mixed_transition
    Cols: eval epochs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_sorted = sorted(eval_epochs_tuple)
    n_epochs = len(epochs_sorted)

    # Infer temperatures from first available result.
    temps: List[float] = []
    for model_name in ("linear_baseline", "transformer"):
        for regime in ("secure", "invalid_fraction", "single_invalid", "mixed_transition"):
            by_epoch = suite_results.get(model_name, {}).get(regime, {})
            for e in epochs_sorted:
                if e in by_epoch and by_epoch[e]:
                    temps = sorted(by_epoch[e].keys())
                    break
            if temps:
                break
        if temps:
            break
    if not temps:
        return

    row_specs = [
        ("linear_baseline", "secure", "Baseline_secure"),
        ("linear_baseline", "invalid_fraction", "Baseline_insecure1"),
        ("linear_baseline", "single_invalid", "Baseline_insecure2"),
        ("linear_baseline", "mixed_transition", "Baseline_insecure3"),
        ("transformer", "secure", "Transformer_secure"),
        ("transformer", "invalid_fraction", "Transformer_insecure1"),
        ("transformer", "single_invalid", "Transformer_insecure2"),
        ("transformer", "mixed_transition", "Transformer_insecure3"),
    ]

    for temp in temps:
        # Per-model y-range: rows 0..3 (linear) share one range, rows 4..7 (transformer) share another.
        linear_vals: List[np.ndarray] = []
        transformer_vals: List[np.ndarray] = []
        for model_name, regime, _ in row_specs:
            by_epoch = suite_results.get(model_name, {}).get(regime, {})
            for e in epochs_sorted:
                if e in by_epoch and temp in by_epoch[e]:
                    if model_name == "linear_baseline":
                        linear_vals.append(by_epoch[e][temp])
                    else:
                        transformer_vals.append(by_epoch[e][temp])

        def _yrange(vals: List[np.ndarray]) -> Tuple[float, float]:
            if not vals:
                return 0.0, 1.0
            arr = np.concatenate(vals)
            lo, hi = float(arr.min()), float(arr.max())
            if np.isclose(lo, hi):
                hi = lo + 1e-8
            return lo, hi

        linear_y_min, linear_y_max = _yrange(linear_vals)
        transformer_y_min, transformer_y_max = _yrange(transformer_vals)

        fig, axes = plt.subplots(8, n_epochs, figsize=(4 * n_epochs, 16), sharey=False)
        if n_epochs == 1:
            axes = axes.reshape(8, 1)

        for r, (model_name, regime, row_label) in enumerate(row_specs):
            by_epoch = suite_results.get(model_name, {}).get(regime, {})
            for c, epoch in enumerate(epochs_sorted):
                ax = axes[r, c]
                if epoch in by_epoch and temp in by_epoch[epoch]:
                    trace = by_epoch[epoch][temp]
                    taus = np.arange(1, len(trace) + 1)
                    ax.plot(taus, trace, marker="o", ms=2, lw=1.1)
                if r == 0:
                    ax.set_title(f"Epoch E={epoch}")
                if c == 0:
                    ax.set_ylabel(f"{row_label}\ntrace(cov)")
                if r == 7:
                    ax.set_xlabel("tau")
                if model_name == "linear_baseline":
                    ax.set_ylim(linear_y_min, linear_y_max)
                else:
                    ax.set_ylim(transformer_y_min, transformer_y_max)
                ax.grid(True, alpha=0.25)

        fig.suptitle(f"Jailbreak suite comparison | T={temp}")
        fig.tight_layout()
        safe_temp = str(temp).replace(".", "_")
        fig.savefig(output_dir / f"trace_T{safe_temp}.png", dpi=600)
        plt.close(fig)

    print(f"Saved 8-row jailbreak suite plots to: {output_dir}")


def plot_jailbreak_suite_training_losses(
    loss_curves: Dict[str, Tuple[List[float], List[float]]],
    output_path: Path,
) -> None:
    """
    One panel per regime: secure, insecure1, insecure2, insecure3.
    Each panel shows both transformer and baseline training loss.
    """
    regime_order = ["secure", "insecure1", "insecure2", "insecure3"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=False)
    if len(regime_order) == 1:
        axes = [axes]

    for i, regime in enumerate(regime_order):
        ax = axes[i]
        if regime in loss_curves:
            tr_losses, bl_losses = loss_curves[regime]
            epochs_axis = list(range(1, len(tr_losses) + 1))
            ax.plot(epochs_axis, tr_losses, marker="o", ms=3, lw=1.2, label="TinyCausalTransformer", color="C0")
            ax.plot(epochs_axis, bl_losses, marker="s", ms=3, lw=1.2, label="LinearBaseline", color="C1")
        ax.set_title(regime)
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Train CE loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Training loss (cross-entropy): secure vs jailbreak regimes")
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
    poison_rate: float = 0.0,
    poison_mode: str = "partial",
    poison_corruption_rate: float = 0.5,
    eval_prompt: torch.Tensor | None = None,

    transition_primary: np.ndarray | None = None,
    transition_secondary: np.ndarray | None = None,
    secondary_rate: float = 0.0,
) -> Tuple[
    Dict[str, Dict[int, Dict[float, np.ndarray]]],
    Dict[str, Dict[int, Dict[float, np.ndarray]]],
    List[float],
    List[float],
]:
    """
    Train/eval once for a specific (D=vocab_size, q) using an arbitrary row-stochastic transition matrix.
    Training data can be partially poisoned via poison_rate (fraction of invalid sequences).
    poison_mode: "partial" | "single" controls how poison sequences are built.
    Returns:
      transformer_results[mean_method][epoch][temperature] = trace curve
      baseline_results[mean_method][epoch][temperature] = trace curve
      transformer_losses: CE loss per epoch (length = train_epochs)
      baseline_losses:    CE loss per epoch (length = train_epochs)
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    transition = transition_primary
    if transition is None:
        transition = build_sparse_row_stochastic_transition_matrix(vocab_size=vocab_size, q=q, rng=rng)

    dataset = build_dataset(
        cfg,
        rng,
        transition=transition,
        poison_rate=poison_rate,
        poison_mode=poison_mode,
        poison_corruption_rate=poison_corruption_rate,
        secondary_transition=transition_secondary,
        secondary_rate=secondary_rate,
    )
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

    if eval_prompt is None:
        eval_seq = random_walk_sequence_from_transition(transition, cfg.walk_length, rng)
        prompt = torch.tensor(eval_seq[:cfg.prompt_len], dtype=torch.long).unsqueeze(0)
    else:
        prompt = eval_prompt.clone()

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


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    if cfg.prompt_len > cfg.walk_length:
        raise ValueError("prompt_len should be <= walk_length.")
    if cfg.q > cfg.D:
        raise ValueError(f"q must be <= D, got q={cfg.q}, D={cfg.D}.")
    if cfg.poison_rate < 0.0 or cfg.poison_rate > 1.0:
        raise ValueError(f"poison_rate must be in [0,1], got {cfg.poison_rate}.")
    if cfg.jailbreak_invalid_transition_rate < 0.0 or cfg.jailbreak_invalid_transition_rate > 1.0:
        raise ValueError(f"jailbreak_invalid_transition_rate must be in [0,1], got {cfg.jailbreak_invalid_transition_rate}.")
    if cfg.jailbreak_mixed_secondary_rate < 0.0 or cfg.jailbreak_mixed_secondary_rate > 1.0:
        raise ValueError(f"jailbreak_mixed_secondary_rate must be in [0,1], got {cfg.jailbreak_mixed_secondary_rate}.")

    eval_epochs: Tuple[int, ...] = cfg.eval_epochs
    if not eval_epochs:
        eval_epochs = (0,)
        k = 2
        while k <= cfg.train_epochs:
            eval_epochs = (*eval_epochs, k)
            k *= 2
    if max(eval_epochs) > cfg.train_epochs or min(eval_epochs) < 0:
        raise ValueError("All eval_epochs must be in [0, train_epochs].")

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
    out_dir = Path(__file__).resolve().parent / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Jailbreak Suite Experiment ===")
    print(
        f"D={cfg.D}, q={cfg.q}, poison_rate={cfg.poison_rate}, "
        f"partial_corruption_rate={cfg.jailbreak_invalid_transition_rate}, "
        f"mixed_secondary_rate={cfg.jailbreak_mixed_secondary_rate}"
    )
    print(f"temperatures={cfg.temperatures}, eval_epochs={eval_epochs}, effective_steps={effective_steps}, effective_max_lag={effective_max_lag}")
    print(f"device={device}")

    suite_seed = cfg.seed + 90000
    rng_a = np.random.default_rng(suite_seed)
    rng_b = np.random.default_rng(suite_seed + 1)
    transition_a = build_sparse_row_stochastic_transition_matrix(vocab_size=cfg.D, q=cfg.q, rng=rng_a)
    transition_b = build_sparse_row_stochastic_transition_matrix(vocab_size=cfg.D, q=cfg.q, rng=rng_b)

    sim = transition_similarity(transition_a, transition_b, walk_length=cfg.walk_length)
    print("=== Transition matrix similarity (A vs B) ===")
    for k, v in sim.items():
        print(f"  {k}: {v}")
    sim_path = out_dir / "transition_similarity.json"
    with open(sim_path, "w") as f:
        json.dump(sim, f, indent=2)
    print(f"Saved transition similarity to: {sim_path}")

    prompt_rng = np.random.default_rng(suite_seed + 2)
    base_seq = random_walk_sequence_from_transition(transition_a, cfg.walk_length, prompt_rng)
    eval_prompt = torch.tensor(base_seq[: cfg.prompt_len], dtype=torch.long).unsqueeze(0)

    # Regime 0 (secure): fully valid sequences from A, no poison.
    sec_tr, sec_bl, sec_tr_losses, sec_bl_losses = run_single_setting_for_comparison(
        cfg,
        vocab_size=cfg.D,
        q=cfg.q,
        eval_epochs=eval_epochs,
        effective_steps=effective_steps,
        effective_max_lag=effective_max_lag,
        device=device,
        seed=suite_seed + 10,
        transition_primary=transition_a,
        poison_rate=0.0,
        eval_prompt=eval_prompt,
    )
    # Regime 1 (insecure1): poison = valid walks with jailbreak_invalid_transition_rate fraction of transitions corrupted.
    inv_tr, inv_bl, inv_tr_losses, inv_bl_losses = run_single_setting_for_comparison(
        cfg,
        vocab_size=cfg.D,
        q=cfg.q,
        eval_epochs=eval_epochs,
        effective_steps=effective_steps,
        effective_max_lag=effective_max_lag,
        device=device,
        seed=suite_seed + 11,
        transition_primary=transition_a,
        poison_rate=cfg.poison_rate,
        poison_mode="partial",
        poison_corruption_rate=cfg.jailbreak_invalid_transition_rate,
        eval_prompt=eval_prompt,
    )
    # Regime 2 (insecure2): poison = valid walks with exactly one invalid transition.
    one_tr, one_bl, one_tr_losses, one_bl_losses = run_single_setting_for_comparison(
        cfg,
        vocab_size=cfg.D,
        q=cfg.q,
        eval_epochs=eval_epochs,
        effective_steps=effective_steps,
        effective_max_lag=effective_max_lag,
        device=device,
        seed=suite_seed + 12,
        transition_primary=transition_a,
        poison_rate=cfg.poison_rate,
        poison_mode="single",
        eval_prompt=eval_prompt,
    )
    # Regime 3 (insecure3): no poison; (1 - r) valid from A, r valid from B.
    mix_tr, mix_bl, mix_tr_losses, mix_bl_losses = run_single_setting_for_comparison(
        cfg,
        vocab_size=cfg.D,
        q=cfg.q,
        eval_epochs=eval_epochs,
        effective_steps=effective_steps,
        effective_max_lag=effective_max_lag,
        device=device,
        seed=suite_seed + 13,
        transition_primary=transition_a,
        transition_secondary=transition_b,
        secondary_rate=cfg.jailbreak_mixed_secondary_rate,
        poison_rate=0.0,
        eval_prompt=eval_prompt,
    )

    # 8-row comparison plots (one figure per temperature per mean method).
    jailbreak_out_root = Path(__file__).resolve().parent / "result_jailbreak_suite"
    for mean_method in cfg.mean_methods:
        jailbreak_suite_mm = {
            "linear_baseline": {
                "secure": sec_bl[mean_method],
                "invalid_fraction": inv_bl[mean_method],
                "single_invalid": one_bl[mean_method],
                "mixed_transition": mix_bl[mean_method],
            },
            "transformer": {
                "secure": sec_tr[mean_method],
                "invalid_fraction": inv_tr[mean_method],
                "single_invalid": one_tr[mean_method],
                "mixed_transition": mix_tr[mean_method],
            },
        }
        plot_jailbreak_suite_8rows_by_temperature(
            jailbreak_suite_mm,
            jailbreak_out_root / mean_method,
            eval_epochs,
        )

    # Training loss: one panel per regime.
    jailbreak_loss_curves = {
        "secure": (sec_tr_losses, sec_bl_losses),
        "insecure1": (inv_tr_losses, inv_bl_losses),
        "insecure2": (one_tr_losses, one_bl_losses),
        "insecure3": (mix_tr_losses, mix_bl_losses),
    }
    plot_jailbreak_suite_training_losses(jailbreak_loss_curves, out_dir / "training_loss.png")
    print(f"Saved jailbreak suite plots to: {jailbreak_out_root}")
    print(f"Saved training loss plot to: {out_dir / 'training_loss.png'}")


if __name__ == "__main__":
    main()
