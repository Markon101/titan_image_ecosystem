# TITAN Image v9 mathematical notes

This document records implemented equations. Coefficients are configuration
values; tensor shapes and exact defaults are in `src/config.rs`.

## Recurrent world

At developmental step `k`, the world is:

~~~text
W_k = (S_micro,k, S_macro,k, m_k, age, target, active_morph_depth)
~~~

The micro and macro derivatives combine learned NCA, recurrent-interface
writeback, local reference drive, contractive leak, and enabled physical
operators:

~~~text
dS/dt = gain_nca F_NCA(S, context, genome)
      + B_interface
      + fidelity * gain_ref * tanh(P_ref(reference))
      - state_leak * S
      + F_reaction + F_phase + F_cyclic + F_forcing
~~~

Euler and midpoint integration are supported. The proposed state is smoothly
projected:

~~~text
q = S_proposed / limit
project(S_proposed) = S_proposed / (1 + q^4)^(1/4)
~~~

This is odd, smooth for finite inputs, near identity at the origin, and
asymptotic to `+/-limit`. It does not create hard-clamp zero derivatives.

The state soft barrier is:

~~~text
L_state = mean(max(|S|-state_soft_limit, 0)^2)
~~~

with a corresponding memory barrier.

## Recurrent interface and MorphicStack

Pooled micro, macro, reference, genome, fidelity, age, and prior memory enter a
looped shared token processor. The global token stream is capped at 8x8 for
larger interface grids, then written back to local tokens.

After GRU update, active MorphicBlocks apply bounded residuals:

~~~text
m_0 = GRU(token_summary, m_previous)
m_(i+1) = m_i + morph_gain * tanh(C_i swish(E_i Norm(m_i)))
m_next = project_memory(m_active_depth)
~~~

New `C_i` contract weights/biases are zero, so activating a reserved block is
an exact no-op before it learns. Projection occurs once after the active stack.

## Reconstruction++ rendering

The implicit renderer samples micro and macro state at one normalized spatial
view, adds global Fourier coordinates, genome, and bounded LOD, and forms shared
features `h`. Separate heads produce:

~~~text
g = H_ground(h) + state_skip(S_micro, S_macro)
e_raw = emergent_limit * tanh(H_emergent(h))
e = project_emergent(B_low e_raw + B_mid e_raw + B_high e_raw)
z(alpha) = g + alpha e
image(alpha) = Oklab_like_to_RGB(z(alpha))
~~~

The implemented band shaping is:

~~~text
low  = repeated periodic low-pass(e_raw)
mid  = low-pass(e_raw) - low
high = e_raw - low-pass(e_raw)
e_shaped = low_budget*low + mid_budget*mid + high
~~~

At `alpha=0`, `z=g` exactly. The renderer has no raw-reference input.

## Grounding and role objectives

For 2x area pooling `P`:

~~~text
L_fine   = mean |g-y|
L_mid    = mean |P(g)-P(y)|
L_coarse = mean |P(P(g))-P(P(y))|

L_ground = w_fine L_fine + w_mid L_mid + w_coarse L_coarse
         + 0.2 L_ssim_like
~~~

The normal endpoint objective still exposes composite content, palette,
target-aligned signed-gradient L1 structure, boundary-aware seam, and gamut
terms. Texture mode deliberately retains translation-invariant gradient
statistics instead.

The residual diagnostics/regularizers are:

~~~text
actual_effect  = image(alpha) - grounded_image
desired_effect = target - stop_gradient(grounded_image)
L_emergent_fit = mean |actual_effect-desired_effect|
L_emergent_low = mean(P(P(e))^2)
L_emergent_tv  = mean spatial total variation(e)
L_redundancy   = |corr(stop_gradient(g), e)|
~~~

All emergent regularizers are multiplied by the active emergence schedule.

## Developmental curriculum

With normalized age `a`:

~~~text
r = clamp((a - emergence_start) / emergence_ramp, 0, 1)
h = r^2 (3 - 2r)
grounding_schedule = grounding_strength * (1-(1-grounding_floor)h)
emergence_schedule = emergence_strength * h
~~~

## Coordinate-consistent detail

A spatial view is `(x0, y0, size, zoom)` in normalized phenotype coordinates.
For output pixel `(i,j)` at resolution `R`:

~~~text
x = x0 + size * (i+1/2)/R
y = y0 + size * (j+1/2)/R
LOD = log2(max(zoom,1))/5
~~~

The same coordinates select the recurrent-field region and Fourier features.
Source crops use the recorded center-square source transform. Lanczos target
construction supplies antialiased supervision.

For high render `y_2R` and low render `y_R`:

~~~text
L_xres = |P(y_2R)-y_R|_1
       + 0.5 |P(P(y_2R))-P(y_R)|_1
       + 0.25 |edge(P(y_2R))-edge(y_R)|_1
~~~

The light default weight preserves anatomy while allowing zero-mean subpixel
detail at the higher resolution.

## Conditional rectified flow

The experimental flow endpoint is a fixed normalized Oklab transform `x1` of
the global target. With deterministic `epsilon ~ Normal(0,I)` and
`t ~ Uniform(0,1)`:

~~~text
x_t = (1-t) epsilon + t x1
u_t = x1 - epsilon
L_CFM = mean((v_theta(x_t,t,c)-u_t)^2)
~~~

`c` is formed only from sampled recurrent micro/macro fields, normalized
interface memory, age, fidelity, emergence, and LOD. The velocity head receives
no raw reference, target, genome, or Reconstruction++ head output.

Hybrid training is:

~~~text
L_hybrid = endpoint_weight * L_Reconstruction++ + flow_weight * L_CFM
~~~

Frozen analysis sampling solves:

~~~text
dx/dt = v_theta(x,t,c), x(0)=epsilon
~~~

with fixed-step midpoint integration. ODE states are not clamped; nonfinite or
runaway trajectories abort.

## What the equations do not prove

Bounded nonlinear recurrence can exhibit complex long-horizon behavior, but a
finite rollout cannot establish a strange attractor, metaphysical strong
emergence, homeostasis, or causal hierarchy. v9 records recurrence,
perturbation recovery, target separation, ablation, and persistence proxies so
those claims can be evaluated conservatively.
