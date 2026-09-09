"""Compare tensor bytes while excluding only fork-specific checkpoint identity scalars."""
import hashlib
import json
from pathlib import Path
import struct


def tensor_identity(path):
    data=Path(path).read_bytes()
    header_size=struct.unpack('<Q',data[:8])[0]
    header=json.loads(data[8:8+header_size])
    offset=8+header_size
    result={}
    for name,tensor in header.items():
        if name=='__metadata__' or name.endswith(('.checkpoint_id','.immutable_signature','.resolved_signature')):
            continue
        start,end=tensor['data_offsets']
        assert 0<=start<=end<=len(data)-offset
        result[name]=dict(dtype=tensor['dtype'],shape=tensor['shape'],
                          sha256=hashlib.sha256(data[offset+start:offset+end]).hexdigest())
    return result
