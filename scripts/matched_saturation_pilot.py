#!/usr/bin/env python3
"""Short RMSNorm-only versus RMSNorm+penalty experiment from one preserved parent."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time
from frozen_fork_followup import checkpoint, digest, identities, read, write, verify_evaluation
from emergence_experiments import budget_steps
from checkpoint_tensor_identity import tensor_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-metadata',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--core-updates',type=int,default=32)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    assert 1<=args.core_updates<=32
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    parent=read(args.parent_metadata)['config']
    protected=identities(checkpoint(parent)+[args.parent_metadata])
    manifest=read(checkpoint(parent)[-1])
    steps=budget_steps(manifest['world_step'],parent['bptt'],parent['core_update_every'],args.core_updates)
    binary=Path('target/release/titan_image').resolve()
    env=dict(os.environ,OCL_ICD_ASSUME_ICD_EXTENSION='1',RAYON_NUM_THREADS='8')
    configs={}
    jobs=[]
    for name,penalty in [('control',None),('penalty',dict(weight=.0001,threshold=2.5))]:
        config=copy.deepcopy(parent)
        config.update(output_dir=str(root/name),run_tag='matched-'+name,steps=steps,threads=8,
            compute_backend='opencl',terminal='quiet',gallery=0,output_resolution=192,snapshot_resolution=192,
            snapshot_every=0,checkpoint_every=0,log_every=4,save_state_atlas=False,fresh=False,render_only=False)
        config['detail']['cache_dir']=str(root/name/'cache')
        config['analysis'].update(only=False,render_attribution=False,model_stats=False,autonomous_horizon=0,
            perturbation_horizon=0,dynamics_horizon=0,emergence_gallery=False,benchmark=False,probe_dir=None,compare_v8_dir=None)
        config['experiment']=dict(norm='differentiable',saturation_penalty=penalty,withdrawal=None,
                                  optimizer_diagnostics=True,panel=None)
        request=dict(parent_metadata=str(args.parent_metadata.resolve()),destination=config,
                     optimizer='retain',world='retain',warmup='retain')
        write(root/f'{name}-request.json',request)
        configs[name]=config
        jobs.append((name+'-fork',['fork',str(root/f'{name}-request.json')],None))
    for name,config in configs.items():
        jobs.append((name+'-training',['--config-json',str(root/name/'config.json')],None))
        evaluation=copy.deepcopy(config)
        evaluation['analysis']['only']=True
        evaluation['experiment']['panel']=dict(targets=[0],seeds=[42],burn_in=64,diagnostic_ages=[0,20,64],
            horizons=[128,256,512],stride=32,recovery_horizon=512,recovery_cases=['macro_noise','macro_patch'],
            history_capacity=17,clock_robustness=False)
        write(root/f'{name}-evaluation.json',evaluation)
        jobs.append((name+'-evaluation',['--config-json',str(root/f'{name}-evaluation.json')],evaluation))
    write(root/'plan.json',dict(parent_manifest=manifest,parent_hashes=protected,binary_sha256=digest(binary),
        matched_full_core_updates=args.core_updates,development_steps=steps,
        configurations=configs,jobs=[dict(label=l,command=c) for l,c,_ in jobs],
        interpretation='Short causal intervention test; no attribution of the longer historical fork effect.'))
    if not args.execute:
        print('Prepared',root);return
    receipts={}
    try:
        # Import both arms before training and prove identical learned starting tensors.
        initial={}
        for label,command,evaluation in jobs:
            name=label.split('-')[0]
            print('START',label,flush=True);start=time.monotonic()
            before=identities(checkpoint(configs[name])) if evaluation else None
            with (root/f'{label}.log').open('x') as log:
                try:
                    proc=subprocess.run([str(binary),*command],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1800)
                    code=proc.returncode
                except subprocess.TimeoutExpired:
                    code='timeout'
            receipt=dict(returncode=code,seconds=time.monotonic()-start,log_sha256=digest(root/f'{label}.log'))
            if code==0:
                if label.endswith('-fork'):
                    initial[name]=[tensor_identity(p) for p in checkpoint(configs[name])[:3]]
                    if len(initial)==2:
                        assert initial['control']==initial['penalty'],'Unequal starting model/world/optimizer'
                        write(root/'initial-parity.json',dict(identical=True,hashes=initial))
                elif label.endswith('-training'):
                    metadata=read(next((root/name).glob('titan_image_run_metadata*.json')))
                    receipt.update(full_core_windows=metadata['full_core_windows'],development_steps=metadata['completed_development_steps'])
                    assert receipt['full_core_windows']==args.core_updates and receipt['development_steps']==steps
                else:
                    receipt.update(verify_evaluation(evaluation))
                    assert identities(before)==before
                    receipt['checkpoint_unchanged']=True
            receipt['parent_unchanged']=identities(protected)==protected
            write(root/f'{label}-receipt.json',receipt);receipts[label]=receipt
            assert code==0 and receipt['parent_unchanged'],label
            print('DONE',label,round(receipt['seconds'],2),flush=True)
        write(root/'completed.json',dict(receipts=receipts))
    finally:
        after=identities(protected)
        write(root/'preservation.json',dict(identical=after==protected,after=after))
        assert after==protected


if __name__=='__main__':
    main()
