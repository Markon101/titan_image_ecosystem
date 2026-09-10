#!/usr/bin/env python3
"""Run bounded, separate frozen ablations against an existing compatible checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
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


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return True


def validate(root):
    manifest = read(root / 'manifest.json')
    summary = read(root / 'summary.json')
    assert summary['complete'] and summary['training_files_unchanged']
    for path, expected in summary['artifacts_sha256'].items():
        assert digest(path) == expected, path
    rows = [json.loads(x) for x in (root / 'trajectory.jsonl').read_text().splitlines()]
    responses = [json.loads(x) for x in (root / 'response.jsonl').read_text().splitlines()]
    assert finite(rows) and finite(responses) and finite(summary)
    assert manifest['conditioning']['references_present'] is False
    assert manifest['conditioning']['reference_fidelity'] == 0
    for row in rows:
        for name in ('micro', 'macro'):
            field = row[name]
            _, _, h, w = field['shape']
            assert math.isclose(sum(field['spectral_energy']), field['energy_per_site'] * 2*h*w, rel_tol=1e-9)
        if row['update_l2'] is not None:
            assert math.isclose(row['update_l2'], row['update_components']['actual_update_l2'], rel_tol=1e-12)
    recurrence = read(root / 'recurrence.json')
    n = len(recurrence['developmental_ages'])
    raw = (root / 'recurrence.f64le').read_bytes()
    assert len(raw) == n*n*8
    matrix = struct.unpack('<' + 'd' * (n*n), raw)
    assert all(math.isfinite(x) and x >= 0 for x in matrix)
    assert all(matrix[i*n+i] == 0 for i in range(n))
    assert all(matrix[i*n+j] == matrix[j*n+i] for i in range(n) for j in range(n))
    return dict(summary=summary, initial=rows[0], final=rows[-1], responses=responses,
                recurrence_initial_to_final=matrix[n-1]), rows


def execute(command, log, timeout):
    started = time.monotonic()
    peak_kib = 0
    with log.open('x') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError('controlled run exceeded timeout')
                try:
                    for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                        if line.startswith('VmHWM:'):
                            peak_kib = max(peak_kib, int(line.split()[1]))
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
            assert proc.returncode == 0, f'run failed; see {log}'
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return dict(wall_seconds=time.monotonic()-started,
                sampled_process_high_water_rss_kib=peak_kib or None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path('target/release/titan_develop'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--timeout', type=int, default=900, help='per-run timeout seconds')
    args = p.parse_args()
    assert 1 <= args.steps <= 64, 'smoke test horizon must be 1..64'
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    config_path = args.config.resolve()
    original = read(config_path)
    config = original.get('config', original)
    source = Path(config['output_dir']).resolve()
    # Audit all existing files at the checkpoint directory's top level, including
    # legacy outputs, metadata and CSVs, beyond the binary's protected-file set.
    protected = {str(x): digest(x) for x in source.iterdir() if x.is_file()}
    protected[str(config_path)] = digest(config_path)
    write(root / 'protected_before.json', protected)
    fixtures = {
        'off': {},
        'transport': {'transport_max': .01, 'velocity_x': {'terms': [[0, 1.0]]},
                      'velocity_y': {'terms': [[1, 1.0]]}},
        'diffusion': {'diffusion_max': .01, 'diffusivity': {'bias': 0.0}},
    }
    for name, fixture in fixtures.items():
        write(root / f'{name}.json', fixture)
    common = [str(args.binary.resolve()), '--config', str(config_path),
              '--steps', str(args.steps), '--perturb-epsilon', '.02', '--seed', '42']
    results = {}
    baseline_rows = None
    for name in ('baseline', 'off', 'transport', 'diffusion'):
        out = root / name
        command = common + ['--output', str(out)]
        if name == 'baseline':
            command += ['--response-ages', f'0,{args.steps}',
                        '--response-epsilons', '.02,.04', '--response-bands', 'low,mid,high']
        else:
            command += ['--operators', str(root / f'{name}.json')]
        write(root / f'{name}_command.json', command)
        print(f'Starting {name}: {out}', flush=True)
        measured = execute(command, root / f'{name}.log', args.timeout)
        result, rows = validate(out)
        result.update(measured)
        results[name] = result
        if name == 'baseline':
            baseline_rows = rows
        elif name == 'off':
            assert rows == baseline_rows, 'zero sidecar changed trajectory'
            assert digest(root / 'baseline' / 'recurrence.f64le') == digest(out / 'recurrence.f64le')
        print(f'Completed {name}: {measured["wall_seconds"]:.2f}s', flush=True)
    # Occupied output must fail, preserving all existing analysis artifacts.
    sentinel = digest(root / 'off' / 'summary.json')
    refused = subprocess.run(common + ['--output', str(root / 'off')], capture_output=True, text=True)
    assert refused.returncode != 0
    assert digest(root / 'off' / 'summary.json') == sentinel
    assert {str(x): digest(x) for x in source.iterdir() if x.is_file()} == {
        k: v for k, v in protected.items() if Path(k).parent == source}
    assert all(digest(path) == expected for path, expected in protected.items())
    write(root / 'validation.json', dict(schema='titan.development.smoke.v1',
        binary_sha256=digest(args.binary), config_sha256=digest(config_path),
        all_source_files_unchanged=True, zero_sidecar_trajectory_records_exact=True,
        occupied_output_refused=True, results=results))
    print(f'Validated artifacts: {root / "validation.json"}', flush=True)


if __name__ == '__main__':
    main()
