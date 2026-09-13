"""Safety gates must survive optimized Python, including tensor byte bounds."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PreparationSafetyTests(unittest.TestCase):
    def test_identity_failure_and_invalid_tensor_bounds_survive_optimization(self):
        code = '''
import json, struct, tempfile
from pathlib import Path
from fresh_norm_pair import require
from checkpoint_tensor_identity import tensor_identity
try:
    require({'weight': 'changed'} == {'weight': 'original'}, 'tensor identity mismatch')
except RuntimeError as error:
    if 'tensor identity mismatch' not in str(error):
        raise
else:
    raise RuntimeError('identity gate was disabled')
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'bad.safetensors'
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header)
    try:
        tensor_identity(path)
    except ValueError as error:
        if 'invalid data offsets' not in str(error):
            raise
    else:
        raise RuntimeError('truncated tensor accepted')
'''
        for flags in ([], ['-O']):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, *flags, '-c', code],
                    cwd=Path(__file__).resolve().parent,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
