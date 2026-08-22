# TITAN Image v8 recurrent-stability record

## Why v8 exists

The v7 `morphic-rin-v7-02` pure-NCA run isolated a recurrent failure without
reaction-diffusion, phase, cyclic, fractal, or quasiperiodic operators. Its
micro field reached 16.7% near-bound occupancy by development step 12, 50% by
step 20, and 100% by step 316. Late rows had zero micro movement, zero spatial
RGB variance, zero edge energy, a cyan output near `(0.0019, 1, 1)`, and loss
near 44. Recurrent-memory RMS exceeded 220 within episodes.

The earlier alien-fluid v7 run independently reached 100% micro and macro
near-bound occupancy while memory RMS exceeded 300. Its renderer sometimes
continued producing structured images. This demonstrated that coordinates and
the content-keyed genome can mask a frozen organism; rendered appearance alone
is not evidence of healthy dynamics.

The old aggregate `image_variance` also reported about 0.221 for a uniform
cyan image because it mixed differences between channel means with spatial
variation. v8 defines this metric as mean within-channel spatial variance.

## Root causes addressed

1. The deterministic initializer overwrote nominally zero NCA and interface
   write heads and overwrote normalization scales with random values.
2. GRU output was bounded, but each morphic residual block made memory
   unbounded and raw memory was fed back into tokens.
3. Interface write heads were linear and therefore had no magnitude bound.
4. Field integration ended in a hard clamp, creating exact saturation and dead
   gradient paths.
5. One core window was followed by three decoder-only windows. Updated
   dynamics could therefore evolve for twelve detached development steps
   before receiving another core correction.
6. The objective had no state or memory stability term.
7. Global gradient telemetry could not distinguish a dead core from an active
   renderer.
8. There was no sustained-saturation safe stop.

## v8 controls

The recurrent interface now uses identity-like initialization:

- NCA output, interface write, attention output, feed-forward contract, and
  morphic contract weights start at zero;
- every bias starts at zero;
- normalization gains start at one;
- remaining matrices retain deterministic variance-scaled initialization.

Morphic memory is smooth-bounded after the GRU and after every active residual
block. Morphic residual gain defaults to 0.02 rather than 0.30, write grids pass
through `tanh`, and episode memory reset is independent and complete by
default.

Dense fields use a quartic smooth projection with nonzero derivative for every
finite input. Stable defaults are `dt=0.12`, `nca_gain=0.25`,
`state_leak=0.10`, gradient clip 1.0, warmup 48 updates, and balanced/fast
core cadence 2. Soft state and memory barriers penalize excess energy without
driving the entire organism toward zero.

A configurable watchdog counts consecutive windows whose micro or macro
near-bound occupancy exceeds 25%. At eight windows it safely stops, flushes
metrics, saves an ordinary checkpoint, marks metadata, writes final
diagnostics, and avoids further gallery development.

## Evidence required before stability claims

Static bounds do not prove that trained dynamics remain useful. A release
claim requires:

- locked unit tests and strict locked Clippy;
- a native locked release build;
- the 256-step recurrent-interface bound test;
- the 256-step autonomous full-core bound test;
- a fresh training smoke run and exact checkpoint resume;
- CSV header/row consistency with finite values;
- nonzero core gradient RMS on full-core rows;
- low near-bound occupancy and bounded memory;
- nonzero within-channel image variance and edge energy after initial warmup;
- matched interface-disabled and enabled runs for causal attribution;
- longer multi-seed family runs before aesthetic or generalization claims.

The watchdog is damage containment, not evidence that the operating point is
optimal. Likewise, a bounded trajectory can still collapse to a constant or a
short cycle. Long-run evaluation must retain spatial diversity, temporal
movement, target-relative quality, null-reference quality, and seed diversity.

## Migration boundary

v8 model, world, optimizer, metrics, and manifest names use the
`titan_image_*_v8` prefix under `/sdcard/Download/titan_image_v8`. The schema
and package version are 8 and 0.8.0. v7 checkpoints are intentionally not
imported: their trained parameters and optimizer moments were learned under
different initialization, recurrent projection, objective, cadence, and field
integration semantics.
