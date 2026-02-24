# Simple transformer
- Initialize simple transformer: specify the number of heads, hidden_dimensions as arguments.
- For training, 
  - Create a grid of size nxn, where each cell corresponds to a token. (A token can appear more than once.)
  - Create a bunch of training sequences by doing a random walk on this grid.
  - Train the transformer on these training sequences for E epochs.

Note: Later, we can formulate our own jailbreak, such as by inserting sequences that are invalid according to our grid 

# Compute time-lagged covariance matrix
- Pick an input sequence
- Set the temperature to T>0
- Set number of trials M to small number (like 5 or 10)
- For the chosen input sequence, 
  - For i = 1, ..., M
    - Generate S outputs and record logit vector at every output. 
    - We end up with SxD matrix where D is the dictionary size
- Let L[i,s,j] denote the logit on trial i at the s-th sequence for token j
- Let mu_i = mean logit vector across S outputs for trial i. 
- For tau = 1,...,m
  - Compute 1/(M x (S - tau)) x sum_i sum_{s=tau}^S (L[i,s] - mu_i) (L[i,s-tau] - mu_i)^\top 
  - CHECK: This should be computing the time-lagged covariance using the logit vectors

Note: (1) there are potentially other notions of covariance that are worth exploring, (2) instead of logits, we can use hidden layers/activations. 

# Evaluations / plots
- Plot the trace of the time-lagged covariance matrix for each tau (y axis = trace, tau = x axis)
- Repeat these plots over different temperature T and different training epochs E
- Make sure to normalize the y axis so that all plots have the same y-range

# Baselines
 Create a class LinearBaseline that has the same functions as SimpleTransformer but is a linear dynamical system under the hood (e.g., stores some latent vector x_t, multiplies that by A, adds some noise to get x_t+1, then multiples by some C to get the logit vector)

# Explanation 
- A continuous-time version of what we have been doing is: stochastic differential equation (SDE) where dx = A dt + B dW
- Interestingly, we can derive a relationship between A, tau, and the time-lagged covariance matrix
- In low-dimensional settings, that actually means that you can SOLVE for A given the time-lagged covariance (which is super cool!)
- We are in the high-dimensional setting, and we don't want to learn A, but we want to learn its eigenvalues/other properties. It turns out that we can infer these from the trace of the time-lagged covariance.
- x(t) = exp(\lambda A t) -> eignevalues of A look like a + bi (where i is an imaginary number = sqrt(-1))