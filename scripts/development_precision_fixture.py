#!/usr/bin/env python3
"""Tiny CPU precision control, not an f64 implementation of the Titan model."""
import argparse
from pathlib import Path
import numpy as np
from development_analysis import write, response_slopes, sha


def fixture():
    x=np.full((16,16), .6)
    p=np.where(np.indices(x.shape).sum(axis=0)%2==0, 1., -1.)/16
    amplitudes=[.0001,.0004,.0016,.0064,.0256,.1024,.4096]
    results={}
    for dtype in (np.float32,np.float64):
        def g(v):
            v=v.astype(dtype)
            return np.tanh(v)+dtype(.2)*v*v
        data=[]
        base=g(x).astype(np.float64)
        for epsilon in amplitudes:
            numerator=g(x+epsilon*p).astype(np.float64)+g(x-epsilon*p).astype(np.float64)-2*base
            # For constant x and a checkerboard direction, the even response is DC.
            low=np.full(x.shape,numerator.mean())
            data.append(dict(offset=0,band='high',epsilon=epsilon,
                numerator_full_l2=float(np.linalg.norm(numerator)),
                numerator_low_spatial_l2=float(np.linalg.norm(low)),
                q_spatial_l2=float(np.linalg.norm(low)/(2*epsilon**2))))
        results[np.dtype(dtype).name]=dict(samples=data,slopes=response_slopes(data))
    second_derivative=-2*np.tanh(.6)*(1-np.tanh(.6)**2)+.4
    expected=abs(second_derivative)/32
    assert abs(results['float64']['samples'][0]['q_spatial_l2']/expected-1)<1e-4
    assert results['float64']['slopes']['groups'][0]['resolved_quadratic_windows']
    return dict(schema='titan.precision_fixture.v1',map='tanh(x)+.2*x^2',shape=[16,16],
                x=.6,perturbation='unit-L2 zero-mean checkerboard',analytic_limit_q_l2=expected,
                results=results,scope='analytic CPU precision fixture only; Titan weights and inference remain f32')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=fixture();payload['script_sha256']=sha(__file__)
    write(args.output,payload)
    print(args.output)
