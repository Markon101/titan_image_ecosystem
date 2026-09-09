#!/usr/bin/env python3
"""Small, ordered real-checkpoint A/B; writes only a NEW experiment directory.

No long training by default. Baseline panel runs before either continuation.
All subprocess output goes to regular files, never image encodings.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint(config):
    suffix = '_' + config['run_tag'] if config['run_tag'] else ''
    root = Path(config['output_dir'])
    return {role: digest(root / f'titan_image_{name}_v9{suffix}.{extension}')
            for role, name, extension in [('model', 'model', 'safetensors'),
                ('world', 'world', 'safetensors'), ('optimizer', 'optimizer', 'safetensors'),
                ('manifest', 'checkpoint', 'json')]}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def budget_steps(start, bptt, cadence, core_updates):
    windows = 0
    cores = 0
    while cores < core_updates:
        cores += int((start // bptt + windows) % cadence == 0)
        windows += 1
    return windows * bptt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-metadata', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/titan_image'))
    parser.add_argument('--core-updates', type=int, default=8)
    parser.add_argument('--targets', type=int, nargs='+', default=[0, 41])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    parser.add_argument('--horizons', type=int, nargs='+', default=[128])
    parser.add_argument('--recovery-horizon', type=int, default=32)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.core_updates <= 32:
        parser.error('initial A/B budget must be 1..32 full-core updates')
    parent = json.loads(args.parent_metadata.read_text())['config']
    suffix = '_' + parent['run_tag'] if parent['run_tag'] else ''
    manifest = json.loads((Path(parent['output_dir']) / f'titan_image_checkpoint_v9{suffix}.json').read_text())
    before = checkpoint(parent)
    args.root = args.root.resolve()
    args.binary = args.binary.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    steps = budget_steps(manifest['world_step'], parent['bptt'], parent['core_update_every'], args.core_updates)
    panel = dict(targets=args.targets, seeds=args.seeds, burn_in=64,
                 diagnostic_ages=[0, 20, 64], horizons=args.horizons, stride=32,
                 recovery_horizon=args.recovery_horizon, history_capacity=8,
                 clock_robustness=False)
    configs = {}
    for name, norm in [('baseline', 'legacy'), ('control', 'legacy'), ('rmsnorm', 'differentiable')]:
        config = copy.deepcopy(parent)
        config.update(output_dir=str(args.root / name), run_tag='emergence-' + name,
                      steps=steps, fresh=False, render_only=False, gallery=0,
                      snapshot_every=0, checkpoint_every=0, log_every=4,
                      output_resolution=192, snapshot_resolution=192, terminal='quiet')
        config['analysis'].update(only=False, render_attribution=False, model_stats=False,
            autonomous_horizon=0, perturbation_horizon=0, dynamics_horizon=0,
            emergence_gallery=False, benchmark=False, probe_dir=None, compare_v8_dir=None)
        config['experiment'] = dict(norm=norm, withdrawal=None, optimizer_diagnostics=True, panel=None)
        request = dict(parent_metadata=str(args.parent_metadata.resolve()), destination=config,
                       optimizer='retain', world='retain', warmup='retain')
        write(args.root / f'{name}-request.json', request)
        evaluation = copy.deepcopy(config)
        evaluation['analysis']['only'] = True
        evaluation['experiment']['panel'] = panel
        write(args.root / f'{name}-evaluation.json', evaluation)
        configs[name] = config
    plan = dict(parent_hashes=before, parent_manifest=manifest, panel=panel,
                matched_full_core_updates=args.core_updates, development_steps=steps,
                executable=str(args.binary), executable_sha256=digest(args.binary),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                source_files={str(p): digest(p) for p in sorted(Path('src').rglob('*.rs'))},
                interpretation='small mechanical/retention pilot; no universal quality threshold or emergence claim')
    write(args.root / 'plan.json', plan)
    if not args.execute:
        print(f'Prepared {args.root}; no checkpoints created or training performed.')
        return
    env = dict(os.environ, OCL_ICD_ASSUME_ICD_EXTENSION='1', RAYON_NUM_THREADS=str(parent['threads']))
    receipts = []

    def run(label, command):
        print(label, flush=True)
        started = time.monotonic()
        with (args.root / f'{label}.log').open('x') as log:
            result = subprocess.run([str(args.binary), *map(str, command)], stdout=log, stderr=subprocess.STDOUT, env=env)
        receipt = dict(label=label, command=list(map(str, command)), returncode=result.returncode,
                       elapsed_seconds=time.monotonic()-started,
                       log_sha256=digest(args.root / f'{label}.log'))
        receipts.append(receipt)
        write(args.root / f'{label}-receipt.json', receipt)
        if checkpoint(parent) != before:
            raise RuntimeError('PARENT CHECKPOINT CHANGED')
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {args.root / (label + ".log")}')

    try:
        for name in configs:
            run(f'{name}-fork', ['fork', args.root / f'{name}-request.json'])
        for name in ['control', 'rmsnorm']:
            run(f'{name}-gradients', ['gradient-check', args.root / name / 'config.json'])
        run('baseline-evaluation', ['--config-json', args.root / 'baseline-evaluation.json'])
        for name in ['control', 'rmsnorm']:
            run(f'{name}-training', ['--config-json', args.root / name / 'config.json'])
            metadata = json.loads(next((args.root / name).glob('titan_image_run_metadata*.json')).read_text())
            if metadata['full_core_windows'] != args.core_updates or metadata['completed_development_steps'] != steps:
                raise RuntimeError(f'{name}: training budget mismatch')
            run(f'{name}-evaluation', ['--config-json', args.root / f'{name}-evaluation.json'])
        evaluations = {}
        for name in configs:
            summary = json.loads(next((args.root / name).glob('titan_image_analysis_v9*.json')).read_text())
            archive = Path(summary['provenance']['archive'])
            if json.loads(archive.read_text()) != summary:
                raise RuntimeError('evaluation archive mismatch')
            for path, identity in summary['provenance']['artifacts'].items():
                if digest(path) != identity['sha256']:
                    raise RuntimeError(f'artifact hash mismatch: {path}')
            evaluations[name] = dict(archive=str(archive), sha256=digest(archive),
                                     checkpoint_hashes=checkpoint(configs[name]))
        write(args.root / 'completed.json', dict(evaluations=evaluations, receipts=receipts))
    finally:
        after = checkpoint(parent)
        write(args.root / 'parent-preservation.json', dict(before=before, after=after, identical=before == after))
        if before != after:
            raise RuntimeError('PARENT CHECKPOINT CHANGED')


if __name__ == '__main__':
    main()
