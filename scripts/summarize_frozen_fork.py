#!/usr/bin/env python3
"""Extract auditable, compact data from completed frozen_fork_followup receipts."""
import argparse
import json
from pathlib import Path
import statistics
from frozen_fork_followup import digest, read, finite


def summarize(root):
    plan = read(root / 'plan.json')
    result = dict(version=1, root=str(root), binary_sha256=plan['binary_sha256'],
                  source_commit=plan['source_commit'], checkpoint_manifests=plan['checkpoint_manifests'],
                  complete=(root / 'completed.json').exists(), probes=[], familiar=[],
                  interface=[], recovery=[], trajectories=[], gradients={}, receipts={})
    for job in plan['jobs']:
        label = job['label']
        path = root / f'{label}-receipt.json'
        if not path.exists():
            continue
        receipt = read(path)
        assert receipt['returncode'] == 0
        assert receipt['protected_unchanged'] and receipt['copies_unchanged']
        assert digest(root / f'{label}.log') == receipt['log_sha256']
        result['receipts'][label] = receipt
        arm = label.split('-')[0]
        if label.endswith('gradients'):
            g = read(root / f'{label}.log')
            assert finite(g) and g['optimizer_updates_applied'] == 0
            result['gradients'][arm] = {k:g[k] for k in ['start_step','loss','saturation_penalty_loss','norms','groups']}
            continue
        assert digest(receipt['archive']) == receipt['archive_sha256']
        a = read(receipt['archive'])
        assert finite(a)
        for p, h in a['provenance']['artifacts'].items():
            assert digest(p) == h['sha256'], p
        probe = a.get('natural_image_probe')
        if probe:
            assert probe['weights_frozen'] and probe['optimizer_steps'] == 0
            assert probe['held_out_by_source_bytes'] and probe['target_count'] == 6
            for target in probe['targets']:
                for p in target['points']:
                    if p['reference_fidelity'] == 0:
                        assert p['micro_reference_drive_rms'] == p['macro_reference_drive_rms'] == 0
                    result['probes'].append(dict(arm=arm,name=target['name'],
                        fingerprint=target['source_fingerprint'],fixed_world_seed=probe['fixed_world_seed'],
                        **{k:p[k] for k in ['age','reference_fidelity','raw_l1','edge_l1','coarse_spatial_l1',
                            'micro_near_bound_fraction','macro_near_bound_fraction']}))
        for p in (a.get('experimental_panel') or {}).get('points', []):
            key = dict(arm=arm,target=p['target'],seed=p['seed'],fingerprint=p['source_fingerprint'])
            result['familiar'].append(dict(**key,phase='guided',offset=0,raw_l1=p['initial_reconstruction']['raw_l1']))
            for row in p['autonomous']:
                result['familiar'].append(dict(**key,phase='autonomous',offset=row['offset'],raw_l1=row['target']['raw_l1']))
            for d in p['interface']:
                for head in ['micro','macro_field']:
                    t=d['trace'][head]
                    result['interface'].append(dict(**key,age=d['age'],fidelity=d['fidelity'],head=head,
                        saturated_fraction=t['saturated_fraction'],mean_tanh_derivative=t['mean_tanh_derivative'],
                        absolute_logit_median=t['absolute_logits']['median'],absolute_logit_p95=t['absolute_logits']['p95'],
                        derivative_below_001=t['derivative_below_001'],spatial_write_variance=t['spatial_write_variance']))
            for rec in p['recovery']:
                initial=rec['initial_state_distance']
                assert initial > 0
                rows=rec['trajectory']
                for row in rows:
                    result['trajectories'].append(dict(**key,mode=rec['mode'],case=rec['case'],offset=row['offset'],
                        ratio=row['state_distance']/initial,macro_updates=row['macro_updates'],rendered_l1=row['rendered_l1']))
                result['recovery'].append(dict(**key,mode=rec['mode'],case=rec['case'],
                    initial_state_distance=initial,state_time_to_half=rec['state_time_to_half'],
                    ratio_256=next(x['state_distance']/initial for x in rows if x['offset']==256),
                    ratio_512=rows[-1]['state_distance']/initial,
                    max_ratio_last128=max(x['state_distance']/initial for x in rows if x['offset']>=384),
                    macro_updates=rec['macro_updates'],final_rendered_l1=rows[-1]['rendered_l1']))
    if (root/'preservation.json').exists():
        result['preservation']=read(root/'preservation.json')
    if (root/'training_diagnostics_summary.json').exists():
        result['training_diagnostics']=read(root/'training_diagnostics_summary.json')
    # Require identical identities in each matched evaluation, not just row order.
    for collection,keys in [('probes',['name','age','reference_fidelity']),
                            ('familiar',['target','seed','phase','offset'])]:
        groups={}
        for row in result[collection]:
            groups.setdefault(tuple(row[k] for k in keys),{})[row['arm']]=row
        for rows in groups.values():
            if set(rows)=={'parent','fork'}:
                assert rows['parent']['fingerprint']==rows['fork']['fingerprint']
                if collection=='probes':
                    assert rows['parent']['fixed_world_seed']==rows['fork']['fixed_world_seed']
    if result['complete']:
        assert len(result['receipts']) == len(plan['jobs'])
        assert len(result['probes']) == 48 and len(result['gradients']) == 2
        panel_jobs = len(plan['jobs']) - 4
        assert len(result['recovery']) == panel_jobs * 4
        assert len(result['interface']) == panel_jobs * 12
        assert len(result['trajectories']) == panel_jobs * 64
        assert all(result['preservation'][k] for k in
                   ['protected_unchanged', 'copies_unchanged', 'probe_sources_unchanged'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    data=summarize(args.root.resolve())
    (args.root/'summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print('Completed jobs:',len(data['receipts']),'complete:',data['complete'])
    for arm in ['parent','fork']:
        for age in [8,64]:
            for fidelity in [1,0]:
                rows=[x for x in data['probes'] if x['arm']==arm and x['age']==age and x['reference_fidelity']==fidelity]
                if rows: print(arm,'probes',age,fidelity,'n',len(rows),'mean L1',statistics.mean(x['raw_l1'] for x in rows))
        mature=[x for x in data['interface'] if x['arm']==arm and x['age']==64 and x['fidelity']==1]
        if mature: print(arm,'mature saturation',statistics.mean(x['saturated_fraction'] for x in mature))
        recovery=[x for x in data['recovery'] if x['arm']==arm]
        if recovery: print(arm,'recovery n',len(recovery),'ratio512 median',statistics.median(x['ratio_512'] for x in recovery),
                           'last128 sampled <=half',sum(x['max_ratio_last128']<=.5 for x in recovery))


if __name__=='__main__':
    main()
