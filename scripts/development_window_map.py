#!/usr/bin/env python3
"""Revalidate a completed panel; export corrected RQA and fixed-window phase signatures."""
import argparse
import csv
from pathlib import Path
import numpy as np
from development_analysis import analyze, read, rows, sha, write, rqa, lyapunov_label


def window_rates(records, window):
    count=len(records[0]['finite_time_exponents'])
    cumulative={0:np.zeros(count)}
    result=[]
    for r in records:
        stop=r['offset']
        cumulative[stop]=np.array(r['finite_time_exponents'])*stop
        start=max(t for t in cumulative if t<=max(0,stop-window))
        rate=(cumulative[stop]-cumulative[start])/(stop-start)
        result.append(dict(offset=stop,start_offset=start,window_steps=stop-start,
                           exponents=rate,regime=lyapunov_label(rate.tolist()),
                           maximum_sampled_exponent=float(rate.max()),
                           basis_policy='QR basis carried from checkpoint; no independent reinitialization at window start'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=int,default=16)
    a=p.parse_args();assert a.window>=4
    source=a.panel.resolve();output=a.output.resolve()
    panel=read(source/'summary.json');assert panel['complete']
    protected={str(source/'summary.json'):sha(source/'summary.json'),str(source/'phase_map.json'):sha(source/'phase_map.json')}
    assert protected[str(source/'phase_map.json')]==panel['phase_map_sha256']
    output.mkdir(exist_ok=False)
    all_points=[];analyses={};rates={}
    for name in panel['results']:
        raw=source/name
        post=analyze(raw,output/name)
        analyses[name]=read(post/'summary.json')
        manifest=read(raw/'manifest.json');trajectory=rows(raw/'trajectory.jsonl')
        ages=np.array(read(raw/'recurrence.json')['developmental_ages'])
        nscalar=sum(np.prod(shape) for shape in manifest['shapes'])
        distances=np.fromfile(raw/'recurrence.f64le',dtype='<f8').reshape(len(ages),len(ages))/np.sqrt(nscalar)
        points=window_rates(rows(raw/'lyapunov.jsonl'),a.window)
        rates[name]=points
        for point in points:
            row=trajectory[point['offset']];age=row['developmental_age']
            keep=np.flatnonzero((ages>=age-point['window_steps'])&(ages<=age))
            energy=np.array(row['micro']['spectral_energy'])+np.array(row['macro']['spectral_energy'])
            point.update(run=name,developmental_age=age,operators=manifest['operator_coefficients'],
                         spectral_fractions=energy/energy.sum(),update_l2=row['update_l2'],
                         recurrence=[rqa(distances[np.ix_(keep,keep)],ages[keep],epsilon) for epsilon in [.01,.05,.1,.2]])
            all_points.append(point)
    sensitivity={}
    for name in ('parent','rmsnorm'):
        sensitivity[name]=[dict(offset=x['offset'],window_steps=x['window_steps'],
                           max_absolute_exponent_difference=float(np.max(abs(x['exponents']-y['exponents']))))
                           for x,y in zip(rates[name+'_baseline'],rates[name+'_epsilon_check'])]
    write(output/'window_phase_map.json',dict(schema='titan.window_phase_map.v2.1',window_steps=a.window,
          points=all_points,epsilon_sensitivity=sensitivity,
          limitation='local windows share an aligned basis; these are finite-amplitude, sampled-direction estimates, not a global regime classifier'))
    with (output/'window_phase_map.csv').open('x', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['run','age','start_offset','window_steps','maximum_sampled_exponent','lambda_1','lambda_2','lambda_3','low_fraction','mid_fraction','high_fraction','update_l2','rr_at_0p1','det_at_0p1','lam_at_0p1','sampled_regime'])
        for point in all_points:
            recurrence=next(r for r in point['recurrence'] if r['epsilon_rms']==.1)
            writer.writerow([point['run'],point['developmental_age'],point['start_offset'],point['window_steps'],point['maximum_sampled_exponent'],*point['exponents'],*point['spectral_fractions'],point['update_l2'],recurrence['recurrence_rate'],recurrence['determinism'],recurrence['laminarity'],point['regime']])
    assert all(sha(p)==digest for p,digest in protected.items())
    write(output/'summary.json',dict(schema='titan.corrected_panel.v2.1',complete=True,source_hashes=protected,
          analyses=analyses,window_map_sha256=sha(output/'window_phase_map.json'),window_csv_sha256=sha(output/'window_phase_map.csv'),
          script_sha256=sha(__file__),correction='return-time distributions use forward returns only; excludes artificial splits across the Theiler gap'))
    print(output)


if __name__=='__main__':
    main()
