# TITAN Image v8 mathematical specification

## State hierarchy

The world is W_n = (X_n, M_n, h_n), where X is the micro field, M is the
slower macro field, and h is recurrent-interface memory.

## Recurrent interface

Micro state, macro state, and RGB reference pyramids are pooled to a GxG token
grid. With fidelity r:

Q_X = pool([X, r R_X])

Q_M = pool([M, r R_M]).

Learned projections, genome g, normalized episode age a, and memory h produce:

T_0 = 0.5(P_X Q_X + P_M Q_M) + P_c[g,r,a] + h.

One shared transformer block is applied K times:

A_k = softmax(Q(T_k) K(T_k)^T / sqrt(d)) V(T_k)

T'_k = T_k + 0.25 P_O A_k

T''_k = T'_k + 0.25 P_2 swish(P_1 norm(T'_k)).

The GRU first produces a bounded candidate, then the active morphic prefix uses
small residual gates:

h'_0 = B_H(GRU(mean(T''_k), h_k))

h'_(l+1) = B_H(h'_l + alpha / sqrt(l+1) W_2,l swish(W_1,l norm(h'_l))).

The smooth odd bound is

B_L(x) = x / (1 + (x/L)^4)^(1/4).

It approaches +/-L without a finite-input zero derivative. The spatial
writebacks are independently bounded:

I_X = g_I upsample(tanh(W_X T)),  I_M = g_I upsample(tanh(W_M T)).

Morphic contracts, attention/feed-forward residual outputs, NCA outputs, and
interface write heads start at zero; normalization scales start at one and all
biases at zero. This makes the initial recurrent system identity-like rather
than secretly randomizing nominal zero-output modules.

Attention is confined to G^2 tokens, so its quadratic term is independent of
dense render resolution.

## Local dynamics

The NCA retains identity, Sobel-x, Sobel-y, and Laplacian perception at radius
one and dilation two. Deterministic asynchronous masks are precomputed and
scheduled without host allocation per step.

The derivative is:

F(Z) = F_NCA(Z) + I(Z,h) + F_physical(Z) - lambda Z.

Euler proposes U_(n+1) = Z_n + dt F(Z_n). Midpoint instead uses

k_1 = F(Z_n)

k_2 = F(Z_n + dt k_1/2)

U_(n+1) = Z_n + dt k_2.

Both finish with the smooth projection Z_(n+1) = B_L(U_(n+1)); v8 has no hard
state clamp. At the defaults, active NCA movement is bounded by
dt*g_NCA = 0.12*0.25 = 0.03 before other terms, while restoring movement at the
3.5 state boundary is dt*lambda*3.5 = 0.042.

## Reference conditioning

Generate mode fixes r=0. Reconstruct mode fixes r=r_max. Hybrid mode samples r
uniformly on [r_min,r_max] and independently replaces it with zero at the
configured dropout probability. The scalar r is supplied as input so scaling
cannot be confused with naturally weak reference features.

## Objective boundary

v8 retains the endpoint image objective: aligned pixel L1 for single/family
mode or normalized color/autocorrelation statistics for texture mode, plus
palette, gradient-structure, seam, and gamut terms. It adds soft energy
barriers rather than a zero-seeking global L2 penalty:

L_state = lambda_s [E relu(|X|-s)^2 + 0.5 E relu(|M|-s)^2]

L_memory = lambda_h E relu(|h|-H/2)^2.

These terms are active on full-core windows; detached decoder-only dynamics do
not falsely claim core gradients. This remains reconstruction-conditioned
recurrent development, not score matching, denoising diffusion, or flow
matching.

## Hybrid Muon

For selected interface matrix gradient G, v8 forms momentum and applies
Newton-Schulz iterations to a Frobenius-normalized matrix:

X <- aX + (bXX^T + c(XX^T)^2)X,

with a=3.4445, b=-4.7750, c=2.0315. Other parameters use AdamW. Global clipping
and decoupled weight decay precede both update geometries.

Muon's usefulness in this small recurrent visual model is an experimental
question, not a consequence of the equation.

Global gradient norm must be finite before any update. Muon additionally
rejects a nonfinite direction norm before Newton-Schulz normalization instead of
silently converting it into an arbitrary direction.
