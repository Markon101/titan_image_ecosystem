#!/usr/bin/env python3
"""Run a small reference-free fixed-diffusion recovery panel against an immutable checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from frozen_fork_followup import digest, identities, read, write, finite


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/titan_develop'))
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    parser.add_argument('--timeout', type=int, default=900, help='Per-seed timeout, seconds')
    args = parser.parse_args()
    if not 1 <= args.steps <= 512 or not 1 <= len(args.seeds) <= 4 or len(set(args.seeds)) != len(args.seeds):
        parser.error('steps must be 1..512 and seeds must contain 1..4 unique values')
    if any(seed < 0 or seed >= 2**64 for seed in args.seeds) or args.timeout < 1:
        parser.error('seeds must fit u64 and timeout must be positive')
    config_path = args.config.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    original = read(config_path)
    config = original.get('config', original)
    source = Path(config['output_dir']).resolve(strict=True)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protected = identities([config_path, *[p for p in source.iterdir() if p.is_file()]])
    source_files = identities([Path('Cargo.toml'), Path('Cargo.lock'), Path('build.rs'),
                              *sorted(Path('src').rglob('*.rs')), Path(__file__)])
    binary_hash = digest(binary)
    write(root / 'protected_before.json', protected)
    write(root / 'source_before.json', source_files)
    results = []
    first = None
    for seed in args.seeds:
        out = root / f'seed_{seed}'
        command = [str(binary), '--config', str(config_path), '--output', str(out),
                   '--steps', str(args.steps), '--seed', str(seed), '--recovery', 'true']
        write(root / f'seed_{seed}_command.json', command)
        print(f'Starting seed {seed}: {args.steps} steps, six frozen trajectories', flush=True)
        tick = time.monotonic()
        with (root / f'seed_{seed}.log').open('x') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, RAYON_NUM_THREADS=str(config['threads'])),
                                 timeout=args.timeout)
        require(run.returncode == 0, f'seed {seed} failed; inspect its log')
        summary = read(out / 'summary.json')
        manifest = read(out / 'manifest.json')
        require(summary['complete'] and summary['training_files_unchanged'], 'incomplete recovery')
        require(finite(summary), 'non-finite recovery results')
        require(manifest['conditioning']['references_present'] is False
                and manifest['conditioning']['reference_fidelity'] == 0, 'reference contamination')
        require(summary['recovery']['reference_free_step_calls'] == args.steps * 6, 'step count mismatch')
        require(summary['recovery']['optimizer_updates'] == 0, 'unexpected optimizer updates')
        require(manifest['binary_sha256'] == binary_hash, 'binary identity mismatch')
        require(identities(summary['artifacts_sha256']) == summary['artifacts_sha256'], 'artifact changed')
        rows = [json.loads(line) for line in (out / 'recovery.jsonl').read_text().splitlines()]
        require(finite(rows), 'non-finite recovery rows')
        require(all(row['same_clock_sequence'] and row['runtime_references_present'] is False
                    and row['reference_fidelity'] == 0 for row in rows), 'pairing/reference mismatch')
        for case in ['control', 'macro_noise', 'macro_patch']:
            require(digest(out / f'off_{case}_00000.png') == digest(out / f'fixed_{case}_00000.png'),
                    'arms have different initial renders')
        if first is not None:
            for image in out.glob('off_control_*.png'):
                require(digest(image) == digest(first / image.name), 'undamaged baseline changed across seeds')
            for arm in ['off', 'fixed']:
                image = f'{arm}_macro_patch_{args.steps:05}.png'
                require(digest(out / image) == digest(first / image), 'seed-independent patch changed')
        first = out
        require(identities(protected) == protected, 'protected checkpoint files changed')
        results.append(dict(seed=seed, wall_seconds=time.monotonic()-tick,
                            summary=str(out/'summary.json'), summary_sha256=digest(out/'summary.json'),
                            recovery=summary['recovery']))
        print(f'Completed seed {seed}: {results[-1]["wall_seconds"]:.1f}s', flush=True)
    require(identities(source_files) == source_files and digest(binary) == binary_hash, 'source/binary changed')
    write(root / 'completed.json', dict(complete=True, binary_sha256=binary_hash,
          all_protected_files_unchanged=True, source_files_unchanged=True,
          steps=args.steps, results=results,
          interpretation='One saved state/genome; two noise seeds are not independent trained models. '
                         'Patch repetitions only check reproducibility. Autonomous metrics are not guided reconstruction.'))
    print(f'Validated panel: {root / "completed.json"}', flush=True)


if __name__ == '__main__':
    main()
