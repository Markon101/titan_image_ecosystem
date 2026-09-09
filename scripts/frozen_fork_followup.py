#!/usr/bin/env python3
"""Bounded, single-use frozen parent/fork comparison; never trains or resumes inputs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checkpoint(config):
    suffix = '_' + config['run_tag'] if config['run_tag'] else ''
    return [Path(config['output_dir']) / f'titan_image_{name}_v9{suffix}.{ext}'
            for name, ext in [('model', 'safetensors'), ('world', 'safetensors'),
                              ('optimizer', 'safetensors'), ('checkpoint', 'json')]]


def identities(paths):
    return {str(p): digest(p) for p in paths}


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def verify_evaluation(config):
    root = Path(config['output_dir'])
    report = read(next(root.glob('titan_image_analysis_v9*.json')))
    provenance = report['provenance']
    assert read(provenance['archive']) == report, 'Latest/archive mismatch'
    assert finite(report), 'Non-finite evaluation'
    for path, identity in provenance['artifacts'].items():
        assert digest(path) == identity['sha256'], path
    zero_samples = 0
    for point in (report.get('experimental_panel') or {}).get('points', []):
        for row in point['autonomous']:
            assert row['reference_fidelity'] == row['micro_reference_drive_rms'] == row['macro_reference_drive_rms'] == 0
            assert row['runtime_references_present'] is False
            zero_samples += 1
        for recovery in point['recovery']:
            assert recovery['same_clock_sequence']
            assert recovery['macro_updates'] == config['experiment']['panel']['recovery_horizon'] // config['macro_update_every']
    return dict(archive=provenance['archive'], archive_sha256=digest(provenance['archive']),
                artifact_count=len(provenance['artifacts']), zero_reference_samples=zero_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-metadata', type=Path, required=True)
    parser.add_argument('--fork-metadata', type=Path, required=True)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/titan_image'))
    parser.add_argument('--targets', type=int, nargs='+', default=[0, 41])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    inputs = {'parent': read(args.parent_metadata)['config'], 'fork': read(args.fork_metadata)['config']}
    protected = identities([args.parent_metadata, args.fork_metadata] +
                           [p for c in inputs.values() for p in checkpoint(c)])
    probe_hashes = identities(p for p in sorted(args.probe_dir.iterdir()) if p.is_file())
    configs = {}
    for name, original in inputs.items():
        dst = root / name
        dst.mkdir()
        for path in checkpoint(original):
            shutil.copyfile(path, dst / path.name)
            assert digest(path) == digest(dst / path.name)
        config = copy.deepcopy(original)
        config.update(output_dir=str(dst), output_resolution=192, snapshot_resolution=192,
                      threads=8, terminal='quiet', gallery=0, save_state_atlas=False,
                      fresh=False, render_only=False, compute_backend='opencl')
        config['detail']['cache_dir'] = str(dst / 'cache')
        config['analysis'].update(only=True, render_attribution=False, model_stats=False,
            autonomous_horizon=0, perturbation_horizon=0, dynamics_horizon=0,
            emergence_gallery=False, benchmark=False, compare_v8_dir=None, probe_dir=None)
        config.setdefault('experiment', {})['panel'] = None
        write(dst / 'config.json', config)
        configs[name] = config
    assert identities(protected) == protected, 'Input changed while copying'
    copied = identities(p for c in configs.values() for p in checkpoint(c))
    jobs = []
    for name, config in configs.items():
        probe = copy.deepcopy(config)
        probe['analysis'].update(probe_dir=str(args.probe_dir.resolve()), probe_ages=[8, 64],
                                 probe_reference_fidelities=[1.0, 0.0])
        path = root / f'{name}-probes.json'
        write(path, probe)
        jobs.append((f'{name}-probes', ['--config-json', str(path)], probe))
        jobs.append((f'{name}-gradients', ['gradient-check', str(root / name / 'config.json')], None))
    for target in args.targets:
        for seed in args.seeds:
            for name, config in configs.items():
                panel = copy.deepcopy(config)
                panel['experiment']['panel'] = dict(targets=[target], seeds=[seed], burn_in=64,
                    diagnostic_ages=[0, 20, 64], horizons=[128, 256, 512], stride=32,
                    recovery_horizon=512, recovery_cases=['macro_noise', 'macro_patch'],
                    history_capacity=17, clock_robustness=False)
                label = f'{name}-t{target}-s{seed}'
                path = root / f'{label}.json'
                write(path, panel)
                jobs.append((label, ['--config-json', str(path)], panel))
    write(root / 'plan.json', dict(binary=str(binary), binary_sha256=digest(binary),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_files=identities(sorted(Path('src').rglob('*.rs'))),
        protected=protected, copied=copied, probe_hashes=probe_hashes,
        original_configs=inputs, checkpoint_manifests={n: read(checkpoint(c)[-1]) for n,c in inputs.items()},
        timeout_per_job=args.timeout, jobs=[dict(label=l, command=c) for l,c,_ in jobs],
        training_steps=0, interpretation='Frozen before/after; training duration, normalization and penalty are confounded.'))
    if not args.execute:
        print(f'Prepared copies and configs in {root}; no evaluation or training executed.')
        return
    env = dict(os.environ, OCL_ICD_ASSUME_ICD_EXTENSION='1', RAYON_NUM_THREADS='8')
    try:
        for label, command, config in jobs:
            print('START', label, flush=True)
            start = time.monotonic()
            log_path = root / f'{label}.log'
            with log_path.open('x') as log:
                try:
                    result = subprocess.run([str(binary), *command], stdout=log, stderr=subprocess.STDOUT,
                                            env=env, timeout=args.timeout)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout'
            receipt = dict(returncode=code, seconds=time.monotonic()-start, log_sha256=digest(log_path))
            if code == 0 and config is not None:
                receipt.update(verify_evaluation(config))
            elif code == 0:
                gradient = read(log_path)
                assert finite(gradient) and gradient['optimizer_updates_applied'] == 0
            receipt['protected_unchanged'] = identities(protected) == protected
            receipt['copies_unchanged'] = identities(copied) == copied
            write(root / f'{label}-receipt.json', receipt)
            assert receipt['protected_unchanged'] and receipt['copies_unchanged']
            assert code == 0, f'{label}: {code}'
            print('DONE', label, round(receipt['seconds'], 2), flush=True)
        write(root / 'completed.json', dict(jobs=len(jobs), training_steps=0))
    finally:
        after = identities(protected)
        copy_after = identities(copied)
        probes_after = identities(probe_hashes)
        write(root / 'preservation.json', dict(protected_unchanged=after == protected,
            copies_unchanged=copy_after == copied, probe_sources_unchanged=probes_after == probe_hashes,
            after=after, copy_after=copy_after))
        assert after == protected and copy_after == copied and probes_after == probe_hashes


if __name__ == '__main__':
    main()
