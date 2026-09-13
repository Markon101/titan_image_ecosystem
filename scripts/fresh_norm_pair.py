#!/usr/bin/env python3
"""Create one untrained small model and two verified normalization arms; never train."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
from checkpoint_tensor_identity import tensor_identity
from emergence_experiments import budget_steps
from frozen_fork_followup import checkpoint, digest, identities, read, write


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/titan_image'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--core-updates', type=int, default=128)
    parser.add_argument('--compute-backend', choices=['cpu', 'opencl'], default='opencl')
    parser.add_argument('--write-diagnostics-every', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.core_updates <= 4096 or args.write_diagnostics_every < 0 or args.threads < 1:
        parser.error('core-updates must be 1..4096, diagnostics cadence nonnegative, and threads positive')
    binary = args.binary.resolve(strict=True)
    corpus = args.corpus_dir.resolve(strict=True)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    marker = root / '.pair-incomplete'
    marker.write_text('Preparation incomplete; use completed.json before launching either arm.\n')
    env = dict(os.environ, OCL_ICD_ASSUME_ICD_EXTENSION='1', RAYON_NUM_THREADS=str(args.threads))

    def run(label, command):
        start = time.monotonic()
        with (root / f'{label}.log').open('x') as log:
            try:
                result = subprocess.run([str(binary), *command], stdout=log, stderr=subprocess.STDOUT,
                                        env=env, timeout=600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        write(root / f'{label}-receipt.json', dict(command=command, returncode=code,
              seconds=time.monotonic()-start, log_sha256=digest(root/f'{label}.log')))
        if code != 0:
            raise RuntimeError(f'{label} failed; see {root / (label + ".log")}')

    initial = root / 'initial'
    run('initialize', ['initialize', '--corpus-dir', str(corpus), '--output-dir', str(initial),
        '--run-tag', 'fresh-initial', '--profile', 's25-fast', '--style', 'pure-nca',
        '--compute-backend', args.compute_backend, '--seed', str(args.seed), '--threads', str(args.threads),
        '--training-rmsnorm', 'legacy', '--optimizer-diagnostics',
        '--write-diagnostics-every', str(args.write_diagnostics_every)])
    parent = read(initial / 'config.json')
    metadata = next(initial.glob('titan_image_run_metadata_v9*.json'))
    initial_report = read(metadata)
    require(initial_report['world_step'] == initial_report['optimizer_updates'] == 0,
            'initialization advanced the world or optimizer')
    require(not parent['fresh'] and parent['experiment']['saturation_penalty'] is None,
            'initialization must resume safely with saturation penalty disabled')
    protected = identities(checkpoint(parent) + [metadata, initial / 'config.json'])
    tensors = [tensor_identity(p) for p in checkpoint(parent)[:3]]
    steps = budget_steps(0, parent['bptt'], parent['core_update_every'], args.core_updates)
    result = dict(binary_sha256=digest(binary), initial_hashes=protected,
                  seed=args.seed, matched_full_core_updates=args.core_updates,
                  development_steps=steps, training_steps_applied=0, arms={})
    for name, norm in [('legacy', 'legacy'), ('rmsnorm', 'differentiable')]:
        config = copy.deepcopy(parent)
        config.update(output_dir=str(root / name), run_tag='fresh-' + name, steps=steps)
        config['detail']['cache_dir'] = str(root / name / 'cache')
        config['experiment']['norm'] = norm
        request = dict(parent_metadata=str(metadata), destination=config,
                       optimizer='retain', world='retain', warmup='retain')
        request_path = root / f'{name}-request.json'
        write(request_path, request)
        run(name + '-fork', ['fork', str(request_path)])
        require([tensor_identity(p) for p in checkpoint(config)[:3]] == tensors,
                f'{name}: initial checkpoint tensors differ')
        require(identities(protected) == protected, f'{name}: protected initialization changed')
        result['arms'][name] = dict(config=str(root/name/'config.json'),
            checkpoint_hashes=identities(checkpoint(config)),
            identical_initial_tensor_values=True, command=[str(binary),'--config-json',str(root/name/'config.json')])
    write(root / 'completed.json', result)
    marker.unlink()
    print(f'Prepared {root}: identical initial tensors, zero optimizer updates, no training launched.')
    print(f'Each first training invocation requests {steps} steps / {args.core_updates} full-core updates.')


if __name__ == '__main__':
    main()
