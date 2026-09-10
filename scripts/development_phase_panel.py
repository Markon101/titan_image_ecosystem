#!/usr/bin/env python3
"""Bounded frozen parameter/age panel; no training, repairs, or checkpoint writes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from development_smoke import execute
from development_analysis import analyze, read, rows, sha, write, lyapunov_label


def protected_files(config_paths):
    files = {}
    for path in config_paths:
        config = read(path)
        config = config.get('config', config)
        source = Path(config['output_dir']).resolve()
        for p in source.iterdir():
            if p.is_file():
                files[str(p)] = sha(p)
        files[str(Path(path).resolve())] = sha(path)
    return files


def compare_epsilon(a, b):
    x, y = rows(a/'lyapunov.jsonl'), rows(b/'lyapunov.jsonl')
    assert [r['offset'] for r in x] == [r['offset'] for r in y]
    return [dict(offset=i['offset'], epsilon_a=i['epsilon'], epsilon_b=j['epsilon'],
                 max_exponent_difference=max(abs(a-b) for a,b in zip(i['finite_time_exponents'], j['finite_time_exponents'])),
                 exponents_a=i['finite_time_exponents'], exponents_b=j['finite_time_exponents']) for i,j in zip(x,y)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--rmsnorm', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path('target/release/titan_develop'))
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    assert 32 <= a.steps <= 128 and a.steps%16==0
    root = a.root.resolve(); root.mkdir(exist_ok=False)
    start = time.monotonic()
    protected = protected_files([a.parent, a.rmsnorm])
    write(root/'protected_before.json', protected)
    base_args = [str(a.binary.resolve()), '--steps', str(a.steps), '--extended', 'true',
                 '--lyap-vectors', '3', '--qr-every', '4', '--seed', '42']
    jobs = [('parent_baseline', a.parent, None, '.16', True),
            ('parent_zero', a.parent, {}, '.16', False)]
    for mechanism in ('transport', 'diffusion'):
        for strength in (.005, .02):
            fixture = {mechanism+'_max': strength}
            if mechanism == 'transport':
                fixture.update(velocity_x={'terms':[[0,1.]]}, velocity_y={'terms':[[1,1.]]})
            jobs.append((f'parent_{mechanism}_{str(strength).replace(".","p")}', a.parent, fixture, '.16', False))
    jobs += [('rmsnorm_baseline', a.rmsnorm, None, '.16', True),
             ('parent_epsilon_check', a.parent, None, '.08', False),
             ('rmsnorm_epsilon_check', a.rmsnorm, None, '.08', False)]
    results = {}; phase = []
    for name, config, fixture, epsilon, response in jobs:
        output = root/name
        command = base_args+['--config', str(config.resolve()), '--output', str(output), '--lyap-epsilon', epsilon]
        if fixture is not None:
            operator_file = root/(name+'_operators.json');write(operator_file, fixture)
            command += ['--operators', str(operator_file)]
        if response:
            ages = sorted(set([0, 16, a.steps//2, a.steps]))
            command += ['--response-ages', ','.join(map(str,ages)), '--response-bands', 'low,mid,high',
                        '--response-epsilons', '.02,.04,.08,.16,.32,.64,1.28,2.56']
        write(root/(name+'_command.json'), command)
        print(f'Starting {name}', flush=True)
        runtime = execute(command, root/(name+'.log'), a.timeout)
        post = analyze(output, root/(name+'_analysis'))
        summary, manifest = read(output/'summary.json'), read(output/'manifest.json')
        signatures = read(post/'phase_signature.json')
        for point in signatures:
            point.update(run=name, operators=manifest['operator_coefficients'], checkpoint_step=manifest['training_step'], epsilon=float(epsilon))
            phase.append(point)
        results[name] = dict(runtime=runtime, summary=summary, post_summary=read(post/'summary.json'),
                             final_signature=signatures[-1], response_precision=read(post/'response_precision.json'),
                             dmd=read(post/'dmd.json'), events=read(post/'events.json'))
        if name == 'parent_zero':
            for filename in ['trajectory.jsonl','observables.jsonl','spatial.f32le','lyapunov.jsonl','recurrence.f64le']:
                assert sha(root/'parent_baseline'/filename) == sha(output/filename), f'zero control changed {filename}'
        assert all(sha(p)==digest for p,digest in protected.items()), 'protected input changed'
        print(f'Completed {name}: {runtime["wall_seconds"]:.1f}s, {signatures[-1]["regime"]}', flush=True)
    sensitivities = {name:compare_epsilon(root/(name+'_baseline'), root/(name+'_epsilon_check')) for name in ('parent','rmsnorm')}
    write(root/'phase_map.json', dict(schema='titan.phase_map.v2', points=phase,
          labels='sampled-subspace sign thresholds only; cumulative QR estimates and prefix RQA; no attractor classifier',
          lyapunov_epsilon_sensitivity=sensitivities,
          sidecar_grid='separate transport_max or diffusion_max in {0,.005,.02}; no combined mechanism experiment'))
    write(root/'summary.json', dict(schema='titan.phase_panel.v2', complete=True, results=results,
          source_files_unchanged=True, protected_after={p:sha(p) for p in protected},
          zero_control_files_byte_identical=True, binary_sha256=sha(a.binary),
          scripts_sha256={str(Path(__file__).resolve()):sha(__file__),
                          str(Path(__file__).with_name('development_analysis.py').resolve()):sha(Path(__file__).with_name('development_analysis.py'))},
          phase_map_sha256=sha(root/'phase_map.json'), elapsed_seconds=time.monotonic()-start))
    print(f'Completed panel: {root}', flush=True)


if __name__ == '__main__':
    main()
