# TITAN Image v5: mathematical and experimental specification

This document describes the equations that the v5 program actually evaluates. It separates consequences of those equations from visual hypotheses that must be tested. v5 is intentionally incompatible with v4: grid sizes, channel counts, conditioning, schedules, losses, optimizer state, and checkpoint identity are runtime-defined.

## 1. State, domain, and notation

Both recurrent fields live on a discrete flat torus:

\[
X_n\in[-L,L]^{C\times H_\mu\times H_\mu},\qquad
M_n\in[-L,L]^{C\times H_M\times H_M}.
\]

Here \(X\) is the fast micro field, \(M\) is the slower macro field, \(C\ge12\), and every spatial stencil wraps modulo its grid dimensions. The S25 profiles choose the dimensions; they are not architectural constants. A genome

\[
g\in[-1,1]^G
\]

conditions both recurrent rules and the renderer. Global step \(n\) is append-only. Organism age \(a\) resets at an episode boundary, while checkpoint continuation restores both values exactly.

The full derivative at either scale is additive:

\[
F_\theta(Z;Q,g,n)
=N_\theta(Z,Q,g,n)
+R(Z)+P(Z)+Y(Z)+A_F(Z)+A_Q(Z).
\]

The terms are respectively the learned NCA, reaction-diffusion, complex phase, cyclic chemistry, fractal-target attraction, and quasiperiodic-target attraction. Every explicit module has an independent nonnegative gain and a CLI ablation switch.

## 2. Near/far neural cellular rule

For each channel, v5 applies identity, horizontal Sobel, vertical Sobel, and five-point Laplacian filters at dilation one and dilation two. If

\[
\mathcal P(Z)=[K_{r,j}*_{\mathbb T^2}Z]_{r\in\{1,2\},j\in\{I,x,y,\Delta\}},
\]

then a cell receives \(8C\) perceived features, \(C\) macro-context features, and \(G\) genome features. Its learned update is

\[
h_1=\operatorname{swish}(W_1[\mathcal P(Z),Q,g]+b_1),
\]

\[
h=h_1+\tfrac12\operatorname{swish}(W_2h_1+b_2),
\]

\[
N_\theta=m_n\,\gamma_N\tanh(W_oh+b_o).
\]

The deterministic asynchronous clock \(m_n(x,y)\in\{0,1\}\) has configured activation probability \(p\). Its hash depends on seed, global step, and cell coordinate. Thus the realized trajectory is reproducible, while pathwise translation equivariance is broken by a fixed mask unless the mask is translated with the state. Setting \(p=1\) removes that qualification.

The output projection starts at zero. Therefore the initial learned residual is exactly zero, although its output weights can receive a nonzero gradient immediately. The explicit fields provide the initial developmental prior.

The second perception ring is the image analogue of TITAN Audio's near/far context: local edges and wider organization are exposed separately instead of asking one receptive scale to represent both.

## 3. Multirate integration

The macro field advances when

\[
a\bmod k_M=0,
\]

then its periodic bilinear upsample is supplied to the micro update. The micro field advances every development step. v5 supports Euler

\[
Z_{n+1}=\Pi_{[-L,L]}(Z_n+hF(Z_n))
\]

and explicit midpoint

\[
k_1=F(Z_n),\qquad
k_2=F(Z_n+\tfrac h2k_1),\qquad
Z_{n+1}=\Pi_{[-L,L]}(Z_n+hk_2).
\]

Before projection, Euler has global error \(O(h)\) and midpoint has global error \(O(h^2)\) under the usual smoothness and Lipschitz assumptions. Projection is a safety rail: when it activates, it changes the modeled flow and invalidates an unqualified numerical-accuracy claim. State RMS should therefore eventually be supplemented by a clamp-fraction metric.

Euler is the default because midpoint evaluates the full derivative twice. On a CPU-only phone, that cost is best spent only when an experiment requires reduced integration error.

## 4. Reaction-diffusion subsystem

Channels zero and one are interpreted as \(u,v\):

\[
\dot u=\gamma_R(D_u\Delta u-uv^2+f(1-u)),
\]

\[
\dot v=\gamma_R(D_v\Delta v+uv^2-(f+k)v).
\]

For the five-point toroidal Laplacian,

\[
(\Delta z)_{ij}=z_{i+1,j}+z_{i-1,j}+z_{i,j+1}+z_{i,j-1}-4z_{ij}.
\]

Its sum is zero because each shifted sum is a permutation of the original. Pure diffusion therefore preserves the discrete mean.

Its Fourier eigenvalues lie in \([-8,0]\). Both Euler and explicit midpoint have absolute stability interval \([-2,0]\) on the negative real axis, so a sufficient pure-diffusion condition is

\[
h\gamma_R D\le\tfrac14.
\]

This bound does not prove stability of the nonlinear, coupled, projected system. It is a useful preflight condition for extreme CLI overrides.

## 5. Complex phase subsystem

Channels two and three form \(z=a+ib\). The implemented equation is

\[
\dot z=\gamma_P\left[
\alpha z-(\beta+i\omega)|z|^2z+(d+i\delta)\Delta z
\right].
\]

Equivalently,

\[
\dot a/\gamma_P
=\alpha a+d\Delta a-\delta\Delta b-\beta r^2a+\omega r^2b,
\]

\[
\dot b/\gamma_P
=\alpha b+d\Delta b+\delta\Delta a-\beta r^2b-\omega r^2a,
\qquad r^2=a^2+b^2.
\]

For a spatially uniform isolated field, the phase terms do not change radius and

\[
\dot r=\gamma_P(\alpha r-\beta r^3).
\]

When \(\alpha,\beta>0\), the nonzero radius \(r_*=\sqrt{\alpha/\beta}\) is locally asymptotically stable. This explains a saturation mechanism; it does not prove that the complete generator has an attractor.

## 6. Cyclic three-field chemistry

Channels six through eight form \(a,b,c\):

\[
\dot a/\gamma_Y
=0.055\Delta a+0.55b-0.55c-0.08a-0.025a^3,
\]

with cyclic permutations for \(\dot b\) and \(\dot c\). The antisymmetric linear coupling rotates activity among the species; diffusion organizes it spatially, and linear/cubic damping oppose unbounded amplitude. This is a deliberately inexpensive visual oscillator, not a claimed biochemical model.

## 7. Contractive and quasiperiodic target fields

The fractal target is produced by an affine iterated-function system

\[
w_i(x)=A_ix+b_i,\qquad
\max_i\lVert A_i\rVert_2=0.52<1.
\]

On nonempty compact subsets with Hausdorff distance, its Hutchinson operator is therefore a contraction. Banach's fixed-point theorem gives a unique compact IFS attractor and geometric convergence of ideal set iteration. The finite chaos-game histogram is blurred, centered, and normalized; it is evidence of an IFS-derived target, not proof that a PNG has a particular fractal dimension.

Unlike v4's constant velocity injection, v5 uses stable target attraction in channel four:

\[
A_F(Z)_4=\gamma_F(T_F-Z_4).
\]

For the isolated Euler update, error \(e_n=Z_{4,n}-T_F\) obeys

\[
e_{n+1}=(1-h\gamma_F)e_n.
\]

It contracts exactly when \(0<h\gamma_F<2\). The target is therefore a bounded organizing field rather than a source that necessarily drives the state into its clamp.

Channel five similarly receives

\[
A_Q(Z)_5=\gamma_Q(T_Q-Z_5),
\]

where \(T_Q\) is a normalized sum of sinusoids with irrationally related frequency coefficients. A finite sampled tensor is periodic by storage and can alias, so the honest term is a quasiperiodic forcing proxy.

## 8. Episode resets and corpus scheduling

At a target change, the carried state \(Z\) and deterministic seed state \(S_e\) are blended:

\[
Z'=(1-\rho)\operatorname{detach}(Z)+\rho S_e,
\qquad 0\le\rho\le1.
\]

This convex reset preserves boundedness when both inputs are bounded. It also breaks the accidental assumption that unrelated corpus images should be pixel-aligned along one uninterrupted trajectory.

In family and texture modes, each epoch is a deterministic content-keyed permutation. Every source appears once before the next epoch. Genomes and corpus identity derive from file bytes rather than paths, so harmless renames do not silently change conditioning. Duplicate bytes remain distinguishable only through a deterministic path tie-break.

The modes make different claims:

- 'single': spatial L1 can learn one aligned image.
- 'family': spatial L1 learns a genome-conditioned aligned family.
- 'texture': normalized statistics remove the false requirement that heterogeneous samples align pixel for pixel.

## 9. Periodic implicit renderer

Micro and macro fields are periodically bilinearly interpolated to render resolution. Interpolation indices and weights are cached. The renderer input at point \(p\) is

\[
q(p)=[\widetilde X(p),\widetilde M(p),c(p),g],
\]

where the coordinate map contains only canvas-scale octave bands

\[
c_B(x,y)=
[\sin(2\pi2^bx),\cos(2\pi2^bx),
  \sin(2\pi2^by),\cos(2\pi2^by)]_{b=0}^{B-1}.
\]

v4 supplied a unit-amplitude coordinate band at the cell-grid frequency, creating an easy visible lattice shortcut. v5 exposes coordinate gain and keeps it low by profile.

The learned pointwise decoder is a residual Swish MLP:

\[
H_0=\operatorname{swish}(W_0q+b_0),\qquad
H_{j+1}=H_j+\tfrac12\operatorname{swish}(W_jH_j+b_j).
\]

Its three outputs are added to a bounded, parameter-free style projection of actual state channels:

\[
\ell_{raw}=D_\theta(q)+s\tanh(B_{style}[\widetilde X,\widetilde M]).
\]

This direct state path makes morphogenesis observable and makes a coordinate-only renderer less able to satisfy the loss while ignoring the organism. Style presets change both operator gains and this basis, so changing style is not a one-variable operator ablation.

Lightness and chroma are bounded:

\[
L=0.08+0.84\sigma(\ell_0),\quad
a=\chi\tanh(\ell_1),\quad
b=\chi\tanh(\ell_2).
\]

The program applies the standard-form OKLab-to-linear-RGB polynomial, records pre-clip gamut excursion,

\[
E_{gamut}=\operatorname{mean}|r-\operatorname{clip}(r,0,1)|,
\]

then maps clipped RGB through \((r+10^{-6})^{1/\gamma}\). This is an OKLab-inspired display transform, not a color-calibrated pipeline. The raw image is the scientific output; toroidal detail, bloom, and shoulder mastering is a separate presentation transform.

Periodic bilinear interpolation and Fourier coordinates define a continuous function on the ideal torus before raster quantization, although its derivative can jump at cell boundaries. The seam metric compares distinct first and last pixel centers, so it should be small for a smooth periodic image, not identically zero.

## 10. Objective

The endpoint loss is

\[
\mathcal L
=w_cL_c+w_pL_p+w_sL_s+w_eL_{seam}+w_gE_{gamut}.
\]

For single and family modes,

\[
L_c=\operatorname{mean}|I-T|.
\]

Texture mode uses normalized cross-channel correlation plus spatial autocorrelation. For centered, per-channel standardized pixels \(\hat I\),

\[
G(I)=\frac1N\hat I\hat I^\top,
\qquad
L_G=\operatorname{mean}(G(I)-G(T))^2.
\]

For lag \(d\in\{1,2,4,8\}\) and each axis,

\[
A_d(I)_c=
\frac{\mathbb E[\bar I_c(p)\bar I_c(p+d)]}
{\mathbb E[\bar I_c(p)^2]+\epsilon}.
\]

Then

\[
L_c=L_G+0.7\operatorname{mean}_{d,axis,c}(A_d(I)-A_d(T))^2.
\]

Palette loss compares channel means and log standard deviations. Structure loss compares log mean-absolute-gradient and log RMS-gradient in both axes. Log scaling makes ratios of weak and strong contrast visible to the optimizer. These statistics are inexpensive and multi-scale, but they are not a semantic perceptual metric.

## 11. Endpoint BPTT and sparse core training

A training window develops for \(B=\texttt{bptt}\) recurrent steps, renders once at the endpoint, evaluates one loss, and applies one optimizer update. This removes v4's repeated render/loss construction at every recurrent step.

On a full-core window, autograd connects the endpoint through the \(B\)-step recurrent trajectory. On a decoder-only window, NCA weights are detached inside pointwise projections; the core pass is genuinely tape-free, while the renderer remains tracked. If full-core cadence is \(k_C\), approximately one in \(k_C\) windows pays recurrent-graph memory and backward cost.

This is exact truncated BPTT for the configured endpoint and horizon. It is not the gradient of an infinite trajectory, and sparse full-core updates trade dynamical adaptation for phone throughput.

## 12. Persistent AdamW

For global gradient vector \(g\), v5 records its norm and applies

\[
\tilde g=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
\]

when clipping threshold \(c>0\). Adam moments and update count are checkpointed. The learning rate uses persisted linear warmup

\[
\eta_t=\eta_{peak}\min(1,t/T_w).
\]

The parameter update is decoupled-weight-decay AdamW with bias-corrected moments. Continuation therefore does not silently reset moments or warmup.

## 13. Logical checkpoint identity

A usable checkpoint is the tuple

\[
(\theta,m,v,X,M,n,a,e,i;\ S,C),
\]

where \(S\) is the fingerprint of every setting that changes evolution or gradients, and \(C\) is the content fingerprint of corpus bytes. Model, optimizer, and world tensors are written atomically; the manifest is published last. Loading rejects missing members, generation disagreement, shape disagreement, schema mismatch, configuration drift, or corpus drift.

This establishes logical continuation of the implemented discrete process. It does not guarantee bitwise equality across different Candle builds, CPU feature sets, thread schedules, or compilers.

## 14. Genome gallery

Gallery genome \(g_v\) interpolates two source genomes and adds a small deterministic mutation:

\[
g_v=\operatorname{clip}((1-\alpha)g_i+\alpha g_j+\epsilon_v,-1,1),
\quad \alpha\in[0.15,0.85],\quad |\epsilon_v|\le0.08.
\]

Every gallery cell starts from a fresh deterministic world and develops for

\[
A_v=A_0+v\,\Delta A
\]

steps. This explores conditioning and developmental age without mutating the trained checkpoint. Visible variation is useful evidence of controllability, not by itself proof of novelty.

## 15. Phone cost model

Ignoring constants and channel/filter details, one recurrent step scales roughly as

\[
O(H_\mu^2C H_{ca}+\mathbf1_M H_M^2C H_{ca}),
\]

while an endpoint render scales roughly as

\[
O(R^2H_r(C+G+B+K H_r)).
\]

Here \(R\) is training resolution and \(K\) the render-block count. Output-only rendering scales with output resolution but does not build a training graph. This explains the v5 phone strategy:

- evolve on compact micro/macro grids;
- render differentiably only once per BPTT window;
- train the recurrent core sparsely;
- cache convolution kernels, coordinates, and interpolation plans;
- cache a bounded number of resized sources;
- render large final images tape-free;
- expose profiles instead of hiding architecture constants.

Thread count is only one variable. Android cpusets and thermal control can reduce available cores or frequency during a sustained run; metadata records requested settings and effective parallelism, while an honest performance experiment must also record thermal conditions.

## 16. What v5 does and does not establish

The implementation establishes deterministic seeded initialization, bounded projected state, toroidal stencils, independently switchable operators, content-bound continuation, and a measurable training/output pipeline.

It does not establish:

- that a low loss is aesthetically good;
- that low movement proves an attractor;
- that an IFS-derived raster has a measured fractal dimension;
- that gallery variation is semantic novelty;
- that mastering improvements came from learned dynamics;
- that one phone timing generalizes across temperatures or Android cpusets;
- that this system is a predictive world model.

An attractor claim requires perturbed trajectories, recurrence or contraction evidence, regeneration tests, boundedness without clamp dominance, and matched ablations. A visual-quality claim requires raw images, mastered images, state atlases, metrics, and preferably blinded human comparison. The audit protocol is defined in METRICS.md.
