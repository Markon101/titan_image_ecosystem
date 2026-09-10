#!/usr/bin/env python3
"""Read-only RQA, DMD, response-convergence, transfer and event analysis (NumPy)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def lines(binary):
    """Lengths of maximal contiguous true runs; include edge-truncated runs."""
    edges = np.diff(np.r_[False, binary, False].astype(int))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def rqa(distance, ages, epsilon, theiler=2, minimum=2):
    ages = np.asarray(ages)
    eligible = abs(ages[:, None] - ages[None, :]) > theiler
    recurrence = (distance <= epsilon) & eligible
    points = int(recurrence.sum())
    diagonals = np.concatenate([lines(np.diag(recurrence, k)) for k in range(-len(ages)+1, len(ages))])
    verticals = np.concatenate([lines(column) for column in recurrence.T])
    dl = diagonals[diagonals >= minimum]
    vl = verticals[verticals >= minimum]
    returns = []
    for reference, row in enumerate(recurrence):
        future = row & (ages > ages[reference] + theiler)
        starts = np.flatnonzero(future & ~np.r_[False, future[:-1]])
        returns.extend(np.diff(ages[starts]).tolist())
    return dict(epsilon_rms=epsilon, theiler_steps=theiler, minimum_line=minimum,
                recurrence_rate=points/int(eligible.sum()) if eligible.any() else None,
                recurrent_points=points, eligible_points=int(eligible.sum()),
                determinism=float(dl.sum())/points if points else None,
                average_diagonal_length=float(dl.mean()) if len(dl) else None,
                maximum_diagonal_length=int(dl.max()) if len(dl) else 0,
                laminarity=float(vl.sum())/points if points else None,
                trapping_time=float(vl.mean()) if len(vl) else None,
                recurrence_time_steps=dict(Counter(returns)),
                recurrence_time_definition='gaps between forward-time entries into recurrent runs after the reference age plus Theiler window; not independent samples',
                line_length_unit='retained samples; edge-truncated lines included')


def response_slopes(data, tolerance=.35):
    groups = {}
    for row in data:
        groups.setdefault((row['offset'], row['band']), []).append(row)
    reports = []
    for (age, band), group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x['epsilon'])
        samples = [dict(epsilon=r['epsilon'], numerator_full_l2=r.get('numerator_full_l2'),
                        numerator_low_l2=r.get('numerator_low_spatial_l2', 2*r['epsilon']**2*r['q_spatial_l2']),
                        q_l2=r['q_spatial_l2']) for r in group]
        intervals = []
        for a, b in zip(samples, samples[1:]):
            if b['epsilon'] <= a['epsilon']:
                continue
            def slope(key):
                if not a[key] or not b[key]:
                    return None
                return np.log(b[key]/a[key])/np.log(b['epsilon']/a['epsilon'])
            full_slope = slope('numerator_full_l2')
            low_slope, q_slope = slope('numerator_low_l2'), slope('q_l2')
            quadratic = (low_slope is not None and abs(low_slope-2) <= tolerance
                         and abs(q_slope) <= tolerance and full_slope is not None
                         and abs(full_slope-2) <= tolerance)
            noise = low_slope is not None and abs(low_slope) <= tolerance and abs(q_slope+2) <= tolerance
            intervals.append(dict(epsilon_low=a['epsilon'], epsilon_high=b['epsilon'],
                                  numerator_full_slope=full_slope, numerator_low_slope=low_slope,
                                  q_slope=q_slope, quadratic_candidate=quadratic, noise_like=noise))
        # A window needs at least two adjacent intervals (three amplitudes), not one slope.
        windows = []
        for start in range(len(intervals)-1):
            if intervals[start]['quadratic_candidate'] and intervals[start+1]['quadratic_candidate']:
                if windows and windows[-1][1] == intervals[start]['epsilon_high']:
                    windows[-1][1] = intervals[start+1]['epsilon_high']
                else:
                    windows.append([intervals[start]['epsilon_low'], intervals[start+1]['epsilon_high']])
        reports.append(dict(offset=age, band=band, samples=samples, intervals=intervals,
                            resolved_quadratic_windows=windows))
    return dict(slope_tolerance=tolerance, minimum_adjacent_intervals=2, groups=reports,
                caveat='quadratic scaling is a numerical resolution check; one direction per band; no causal transfer claim')


def observables(trajectory, extra):
    features = []
    names = ['micro_low', 'micro_mid', 'micro_high', 'macro_low', 'macro_mid', 'macro_high',
             'micro_variance', 'macro_variance', 'memory_l2', 'update_l2']
    if extra:
        names += [f'random_projection_{i}' for i in range(16)]
        names += ['memory_mean', 'memory_std']
    for i, row in enumerate(trajectory):
        # Drop the initial undefined update rather than impute it.
        if row['update_l2'] is None:
            continue
        v = row['micro']['spectral_energy'] + row['macro']['spectral_energy']
        v += [row['micro']['variance'], row['macro']['variance'], row['hidden_memory_l2'], row['update_l2']]
        if extra:
            v += extra[i]['random_projection']
            v += [float(np.mean(extra[i]['memory'])), float(np.std(extra[i]['memory']))]
        features.append(v)
    return np.array(features, dtype=np.float64), names


def dmd(data, rank=6, train_fraction=.7):
    """Training-only scaling/SVD, reduced affine fit and chronological held-out prediction."""
    if len(data) < 16:
        return dict(available=False, reason='need at least 16 observations for held-out fit')
    pairs = len(data)-1
    count = min(pairs-4, max(8, int(pairs*train_fraction)))
    mean = data[:count].mean(axis=0)
    scale = data[:count].std(axis=0)
    active = scale > 1e-12*np.maximum(1, abs(mean))
    if not active.any():
        return dict(available=False, reason='constant observables')
    z = (data[:, active]-mean[active])/scale[active]
    x, y = z[:count].T, z[1:count+1].T
    mx, my = x.mean(axis=1), y.mean(axis=1)
    xc, yc = x-mx[:, None], y-my[:, None]
    u, s, vh = np.linalg.svd(xc, full_matrices=False)
    r = min(rank, int((s > s[0]*1e-8).sum()), count-2)
    if not r:
        return dict(available=False, reason='rank zero')
    u = u[:, :r]
    # Full output map B and reduced K; eig(K) is the reduced observable spectrum.
    b = (yc @ vh[:r].T)/s[:r]
    k = u.T @ b
    def predict(v):
        return (b @ (u.T @ (v-mx).T)).T+my
    train_pred = predict(z[:count])
    test_pred = predict(z[count:-1])
    error = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    train_error = error(train_pred, z[1:count+1])
    test_error = error(test_pred, z[count+1:])
    persistence = error(z[count:-1], z[count+1:])
    rollout = []
    current = z[count].copy()
    for _ in range(pairs-count):
        current = predict(current)
        if not np.isfinite(current).all() or np.linalg.norm(current) > 1e12:
            break
        rollout.append(current.copy())
    eigenvalues = np.linalg.eigvals(k)
    modes = [dict(real=v.real, imaginary=v.imag, magnitude=abs(v), radians_per_step=np.angle(v),
                  label='decaying' if abs(v)<.99 else 'growing' if abs(v)>1.01 else 'persistent_candidate',
                  oscillatory=abs(v.imag)>1e-6) for v in eigenvalues]
    return dict(available=True, method='training-standardized reduced affine DMD; constant offset excluded from eigenspectrum',
                training_pairs=count, test_pairs=pairs-count, rank=r, active_features=np.flatnonzero(active),
                singular_values=s, training_standardized_rmse=train_error,
                test_standardized_rmse=test_error, persistence_standardized_rmse=persistence,
                heldout_skill=1-test_error/persistence if persistence else None,
                recursive_test_rmse=error(np.array(rollout), z[count+1:]) if len(rollout)==pairs-count else None,
                recursive_prediction_completed=len(rollout)==pairs-count,
                compression_relative_error=np.linalg.norm(xc-u@(u.T@xc))/np.linalg.norm(xc),
                eigenvalues=modes, spectral_interpretation_supported=test_error<persistence,
                validation='chronological holdout; thresholds descriptive, not confidence intervals')


def lagged(a, b, max_lag=8):
    result = []
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            x, y = a[:len(a)-lag or None], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b)+lag]
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x)>=4 and np.std(x)>1e-14 and np.std(y)>1e-14 else None
        result.append(dict(lag_steps=lag, correlation=corr, pairs=len(x)))
    return result


def transfer_analysis(trajectory, extra):
    result = {}
    for field_index, name in enumerate(('micro', 'macro')):
        energy = np.array([r[name]['spectral_energy'] for r in trajectory])
        change = np.diff(energy, axis=0)
        correlations = {f'{a}_to_{b}': lagged(change[:, a], change[:, b], min(8, len(change)//3))
                        for a, b in [(0, 1), (1, 2), (0, 2)]}
        result[name] = dict(lagged_energy_changes=correlations,
                            lag_convention='positive lag: first band precedes second; Pearson of energy increments')
        if extra:
            measured = [e['transfer_from_previous'][field_index] for e in extra if e['transfer_from_previous']]
            work = np.array([r['work'] for r in measured])
            residual = np.array([r['budget_residual'] for r in measured])
            result[name].update(work_sum=work.sum(axis=0), positive_work_fraction=(work>0).mean(axis=0),
                                max_absolute_budget_residual=np.max(abs(residual)),
                                identity='E_next-E = 2*<P x,P delta> + ||P delta||^2',
                                scope='net band work of complete update; not conservative inter-band transfer')
    return result


def profile(field, bins=24):
    centered = field-field.mean()
    amplitude = float(abs(centered).max())
    if amplitude <= 1e-12:
        return None
    cy, cx = np.unravel_index(abs(centered).argmax(), centered.shape)
    y, x = np.indices(centered.shape)
    dy, dx = abs(y-cy), abs(x-cx)
    dy, dx = np.minimum(dy, field.shape[0]-dy), np.minimum(dx, field.shape[1]-dx)
    radius2 = dx*dx+dy*dy
    ell = float(np.sqrt((radius2*centered**2).sum()/(centered**2).sum()))
    if ell <= 1e-12:
        return None
    xi = np.sqrt(radius2)/ell
    edges = np.linspace(0, 3, bins+1)
    indices = np.minimum((xi/3*bins).astype(int), bins)
    normalized, counts = [], []
    for i in range(bins):
        mask = indices == i
        counts.append(int(mask.sum()))
        normalized.append(float(centered[mask].mean()/amplitude) if mask.any() else None)
    return dict(amplitude=amplitude, center_xy=[int(cx), int(cy)], ell_cells=ell,
                xi_centers=(edges[:-1]+edges[1:])/2, normalized_profile=normalized, bin_counts=counts,
                definition='periodic radial average of selected micro state channel, mean removed; amplitude max abs; ell energy-weighted RMS radius')


def events(trajectory, spatial, z_threshold=3., window=4, separation=4):
    updates = np.array([r['update_l2'] or 0 for r in trajectory])
    center = np.median(updates[1:]); mad = np.median(abs(updates[1:]-center))
    scale = 1.4826*mad
    if scale <= 1e-12:
        return dict(events=[], collapse_available=False, reason='constant or unresolved update series')
    score = (updates-center)/scale
    candidates = [i for i in range(window, len(updates)-window)
                  if score[i]>=z_threshold and updates[i]>=updates[i-1] and updates[i]>updates[i+1]]
    selected = []
    for i in sorted(candidates, key=lambda i: -updates[i]):
        if all(abs(i-j)>=separation for j in selected):
            selected.append(i)
    records = []
    for i in sorted(selected):
        energy = np.array([r['micro']['spectral_energy'] for r in trajectory[i-window:i+window+1]])
        activity = abs(np.diff(energy, axis=0))
        peak_offsets = activity.argmax(axis=0)+1-window
        record = dict(offset=i, developmental_age=trajectory[i]['developmental_age'], robust_z=score[i],
                      offsets=list(range(-window, window+1)), update_window=updates[i-window:i+window+1],
                      micro_energy_window=energy, band_activity_peak_offsets=peak_offsets,
                      peak_order='L-M-H' if np.all(np.diff(peak_offsets)>0) else 'H-M-L' if np.all(np.diff(peak_offsets)<0) else 'tied_or_mixed')
        if spatial is not None:
            record['profile'] = profile(np.asarray(spatial[i], dtype=np.float64))
        records.append(record)
    profiles = [(e['offset'], e['profile']) for e in records if e.get('profile')]
    collapse = []
    for i, (age, a) in enumerate(profiles):
        for other_age, b in profiles[i+1:]:
            a, b = np.array(a['normalized_profile'], dtype=float), np.array(b['normalized_profile'], dtype=float)
            valid = np.isfinite(a)&np.isfinite(b)
            collapse.append(dict(offsets=[age, other_age], common_bins=int(valid.sum()),
                                 normalized_profile_rmse=float(np.sqrt(np.mean((a[valid]-b[valid])**2))) if valid.any() else None))
            a = profiles[i][1]
    return dict(threshold_median_plus_mad=z_threshold, median_update=center, mad_scale=scale,
                window_steps=window, separation_steps=separation, events=records,
                event_orders=dict(Counter(r['peak_order'] for r in records)),
                collapse_available=len(profiles)>=2, pairwise_collapse=collapse,
                caveat='exploratory radial profiles of one channel; not full multichannel shape similarity or a cascade test')


def lyapunov_label(exponents, tolerance=.01):
    if not exponents:
        return 'unmeasured'
    lo, hi = min(exponents), max(exponents)
    if hi < -.05:
        return 'strongly_contractive_sampled_subspace'
    if lo < -tolerance and hi > tolerance:
        return 'mixed_sampled_subspace'
    if hi > tolerance:
        return 'expanding_sampled_subspace'
    if hi < -tolerance:
        return 'contractive_sampled_subspace'
    return 'near_critical_sampled_subspace'


def local_jacobian(data):
    result = []
    for r in data:
        gram = np.array(r['jv_gram'])
        eigenvalues, vectors = np.linalg.eigh(gram)
        require(eigenvalues.min() >= -1e-10*max(1, eigenvalues.max()), 'invalid Gram matrix')
        gains = np.sqrt(np.maximum(0, eigenvalues))
        result.append(dict(offset=r['offset'], epsilon=r['epsilon'], input_bands=r['input_bands'],
                           directional_jv_norms=np.sqrt(np.maximum(0, np.diag(gram))),
                           restricted_singular_gains=gains,
                           least_amplified_direction_coefficients=vectors[:, 0],
                           greatest_amplified_direction_coefficients=vectors[:, -1],
                           scope='singular values of J restricted to supplied orthonormal micro directions; not global extrema'))
    return result


def analyze(root, output, epsilons=(.01, .05, .1, .2), theiler=2, rank=6, event_z=3.):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    summary, manifest = read(root/'summary.json'), read(root/'manifest.json')
    require(summary['complete'] and summary['training_files_unchanged'], 'incomplete or mutated source run')
    protected = {str(root/'summary.json'): sha(root/'summary.json')}
    for path, digest in summary['artifacts_sha256'].items():
        require(sha(path) == digest, f'artifact hash mismatch: {path}')
        protected[path] = digest
    trajectory = rows(root/'trajectory.jsonl')
    ages = read(root/'recurrence.json')['developmental_ages']
    nscalar = sum(int(np.prod(s)) for s in manifest['shapes'])
    distance = np.fromfile(root/'recurrence.f64le', dtype='<f8').reshape(len(ages), len(ages))/np.sqrt(nscalar)
    require(np.isfinite(distance).all() and np.allclose(distance, distance.T) and np.all(distance.diagonal()==0), 'invalid recurrence matrix')
    extra = rows(root/'observables.jsonl') if (root/'observables.jsonl').exists() else []
    lyap = rows(root/'lyapunov.jsonl') if (root/'lyapunov.jsonl').exists() else []
    local = rows(root/'local_jacobian.jsonl') if (root/'local_jacobian.jsonl').exists() else []
    data, names = observables(trajectory, extra)
    spatial = None
    if (root/'spatial.f32le').exists():
        _, _, h, w = manifest['shapes'][0]
        require((root/'spatial.f32le').stat().st_size == len(trajectory)*h*w*4, 'invalid spatial export size')
        spatial = np.memmap(root/'spatial.f32le', dtype='<f4', mode='r', shape=(len(trajectory), h, w))
    output.mkdir(exist_ok=False)
    response = response_slopes(rows(root/'response.jsonl'))
    rqa_results = [rqa(distance, ages, e, theiler) for e in epsilons]
    dmd_result = dmd(data, rank)
    dmd_result['observable_names'] = names
    # Also fit first/second temporal halves: descriptive stationarity sensitivity.
    dmd_result['temporal_halves'] = [dmd(v, rank) for v in np.array_split(data, 2)]
    transfer = transfer_analysis(trajectory, extra)
    event_result = events(trajectory, spatial, event_z)
    spectra = local_jacobian(local)
    points = []
    for r in lyap:
        age = r['developmental_age']; index = next(i for i, t in enumerate(trajectory) if t['developmental_age']==age)
        energy = np.array(trajectory[index]['micro']['spectral_energy'])+np.array(trajectory[index]['macro']['spectral_energy'])
        retained = np.flatnonzero(np.asarray(ages)<=age)
        points.append(dict(offset=r['offset'], developmental_age=age,
                           finite_time_exponents=r['finite_time_exponents'],
                           regime=lyapunov_label(r['finite_time_exponents']),
                           spectral_fractions=energy/energy.sum(), update_l2=trajectory[index]['update_l2'],
                           recurrence=[rqa(distance[np.ix_(retained, retained)], np.asarray(ages)[retained], e, theiler) for e in epsilons]
                           if len(retained)>2 else []))
    payloads = {'response_precision.json': response, 'recurrence_quantification.json': rqa_results,
                'dmd.json': dmd_result, 'cross_scale.json': transfer, 'events.json': event_result,
                'local_jacobian.json': spectra, 'phase_signature.json': points}
    for name, payload in payloads.items():
        write(output/name, payload)
    require(all(sha(path)==digest for path, digest in protected.items()), 'input analysis changed')
    write(output/'summary.json', dict(schema='titan.dynamical_analysis.v2.1', complete=True,
          source=str(root), source_artifact_hashes=protected, script_sha256=sha(__file__),
          numpy_version=np.__version__, config=dict(rqa_epsilons_rms=epsilons, theiler_steps=theiler, dmd_rank=rank, event_z=event_z),
          artifacts_sha256={str(output/name): sha(output/name) for name in payloads},
          elapsed_seconds=time.monotonic()-started,
          scientific_scope='finite amplitude, finite horizon, sampled directions and observables; no chaos/emergence classification'))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rqa-epsilons', default='.01,.05,.1,.2')
    p.add_argument('--theiler', type=int, default=2)
    p.add_argument('--rank', type=int, default=6)
    p.add_argument('--event-z', type=float, default=3.)
    a = p.parse_args()
    eps = tuple(float(e) for e in a.rqa_epsilons.split(','))
    require(all(np.isfinite(e) and e>0 for e in eps) and a.theiler>=0 and 1<=a.rank<=16 and np.isfinite(a.event_z) and a.event_z>0, 'invalid analysis options')
    print(analyze(a.input, a.output, eps, a.theiler, a.rank, a.event_z))
