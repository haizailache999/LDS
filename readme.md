# Simple transformer
- Initialize simple transformer: specify the number of heads, hidden_dimensions as arguments.
- For training, 
  - Create a grid of size nxn, where each cell corresponds to a token. (A token can appear more than once.)
  - Create a bunch of training sequences by doing a random walk on this grid.
  - Train the transformer on these training sequences for E epochs.
  - We treat our training data as coming from a transition rule (either the grid random walk, or an arbitrary sparse row-stochastic transition matrix P of size D×D with q non-zeros per row).

# Baselines
- Create a class LinearBaseline that has the same functions as the simple transformer.
- Initialize it with latent_dim, input_dim, and vocab_size as arguments; under the hood it is a linear dynamical system: store a latent vector x_t, update x_{t+1} = A x_t + B e(token_t), and output logits_t = C x_t (no process noise). A is scaled to a given spectral radius (such as 0.98) for stability.
- For training, use the same grid, random-walk sequences, and cross-entropy loss as the transformer; train for E epochs (optionally with a separate learning rate and LR schedule for the baseline).

# Jailbreak
- Define a transition i→j as
  - **valid** if P[i,j] > 0 (or j is a grid neighbor of i in the grid setting)
  - **invalid** if P[i,j] = 0 (or j is not a grid neighbor of i in the grid setting)

- **Training and Evaluation** (config: `poison_d`, `poison_q`, `poison_rate`, `jailbreak_mixed_secondary_rate`):
  - **secure (regime 0)**: Train on matrix A with fully valid sequences from A. Evaluate on the shared valid prompt from A.
  - **insecure1 (regime 1)**: Train on matrix A with a fraction `poison_rate` of invalid (poison) sequences and the rest valid from A. The invalid sequences has jailbreak_invalid_transition_rate tokens which are invalid transition. Evaluate on the shared valid prompt from A.
  - **insecure2 (regime 2)**: Train on matrix A with a fraction `poison_rate` of invalid (poison) sequences and the rest valid from A. The invalid sequences has only one token which are invalid transition. Evaluate on the shared valid prompt from A.
  - **insecure3 (regime 3, mixed-transition)**: Train on fraction (1 − r) from A and r from a totally different transition matrix B (r = `jailbreak_mixed_secondary_rate`). Evaluate on the same valid prompt from A.
- **Plots**: Same x/y axes (x = tau, y = trace of time-lagged covariance). For each temperature T, one figure with **8 rows × (#eval epochs) columns**:
  - Rows 1–4: `Baseline_secure`, `Baseline_insecure1`, `Baseline_insecure2`, `Baseline_insecure3`
  - Rows 5–8: `Transformer_secure`, `Transformer_insecure1`, `Transformer_insecure2`, `Transformer_insecure3`
  - Secure and insecure share the same y-range within each model (baseline rows share one y-range; transformer rows share another).

# Compute time-lagged covariance matrix
- Pick an input sequence
- Set the temperature to T>0
- Set number of trials M to small number (like 5 or 10)
- For the chosen input sequence, 
  - For i = 1, ..., M
    - Generate S outputs and record logit vector at every output. 
    - We end up with SxD matrix where D is the dictionary size
- Let L[i,s,j] denote the logit on trial i at step s for token j.
- Center by the per-step mean across trials: μ_s = (1/M) sum_i L[i,s] (mean at step s over trials).
- For tau = 1,...,m
  - Compute 1/(M×(S−tau)) × sum_i sum_{s=tau+1}^S (L[i,s] − μ_s)(L[i,s−tau] − μ_{s−tau})^T

Note: (1) there are potentially other notions of covariance that are worth exploring, (2) instead of logits, we can use hidden layers/activations. 

# Evaluations / plots
- Plot the trace of the time-lagged covariance matrix for each tau (y axis = trace, tau = x axis)
- Repeat these plots over different temperature T and different training epochs E
- Make sure to normalize the y axis so that all plots have the same y-range

# Explanation 
- A continuous-time version of what we have been doing is: stochastic differential equation (SDE) where dx = A dt + B dW
- Interestingly, we can derive a relationship between A, tau, and the time-lagged covariance matrix
- In low-dimensional settings, that actually means that you can SOLVE for A given the time-lagged covariance (which is super cool!)
- We are in the high-dimensional setting, and we don't want to learn A, but we want to learn its eigenvalues/other properties. It turns out that we can infer these from the trace of the time-lagged covariance.
- x(t) = exp(\lambda A t) -> eignevalues of A look like a + bi (where i is an imaginary number = sqrt(-1))
