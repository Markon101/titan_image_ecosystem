# TITAN Image v7 mathematical specification

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

The GRU updates h from token mean, then the active prefix of morphic residual
blocks refines it. A bounded h residual is returned to every token.
Zero-initialized write heads map tokens back to spatial micro/macro biases.

Attention is confined to G^2 tokens, so its quadratic term is independent of
dense render resolution.

## Local dynamics

The NCA retains identity, Sobel-x, Sobel-y, and Laplacian perception at radius
one and dilation two. Deterministic asynchronous masks are precomputed and
scheduled without host allocation per step.

The derivative is:

F(Z) = F_NCA(Z) + F_interface(Z) + F_physical(Z) - lambda Z.

The final term is v7's restoring leak against hard-clamp drift. The hard
projection remains a safety rail, not a claimed dynamical feature.

Euler and midpoint remain available:

Z_(n+1) = clip(Z_n + dt F(Z_n))

or

k_1 = F(Z_n)

k_2 = F(Z_n + dt k_1/2)

Z_(n+1) = clip(Z_n + dt k_2).

## Reference conditioning

Generate mode fixes r=0. Reconstruct mode fixes r=r_max. Hybrid mode samples r
uniformly on [r_min,r_max] and independently replaces it with zero at the
configured dropout probability. The scalar r is supplied as input so scaling
cannot be confused with naturally weak reference features.

## Objective boundary

v7 retains the v6 endpoint objective: aligned pixel L1 for single/family mode
or normalized color/autocorrelation statistics for texture mode, plus palette,
gradient-structure, seam, and gamut terms.

This is reconstruction-conditioned recurrent development. It is not yet a
score-matching, denoising-diffusion, or flow-matching objective.

## Hybrid Muon

For selected interface matrix gradient G, v7 forms momentum and applies
Newton-Schulz iterations to a Frobenius-normalized matrix:

X <- aX + (bXX^T + c(XX^T)^2)X,

with a=3.4445, b=-4.7750, c=2.0315. Other parameters use AdamW. Global clipping
and decoupled weight decay precede both update geometries.

Muon's usefulness in this small recurrent visual model is an experimental
question, not a consequence of the equation.
