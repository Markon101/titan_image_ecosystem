#!/usr/bin/env python3
"""Finalize the bounded follow-up after both runners finish; fail closed on missing evidence."""
import json
from pathlib import Path
import statistics as st
import subprocess
import time
from frozen_fork_followup import read,digest,identities
from summarize_frozen_fork import summarize

R=Path('analysis/rmsnorm_followup_2026-09-09_matched')
P=Path('analysis/saturation_matched_2026-09-09')
start=time.monotonic()
for root in [R,P]:
    while not (root/'completed.json').exists():
        if (root/'preservation.json').exists() or time.monotonic()-start>16000:
            raise RuntimeError(f'Incomplete experiment: {root}; inspect receipts; no final success commit')
        time.sleep(10)
    while not (root/'preservation.json').exists():time.sleep(1)
s=summarize(R.resolve())
assert read(P/'preservation.json')['identical'] and read(P/'initial-parity.json')['identical']
m={}
for arm in ['control','penalty']:
    receipt=read(P/f'{arm}-evaluation-receipt.json')
    assert receipt['returncode']==0 and receipt['checkpoint_unchanged'] and receipt['parent_unchanged']
    assert digest(receipt['archive'])==receipt['archive_sha256']
    a=read(receipt['archive'])
    for p,h in a['provenance']['artifacts'].items():assert digest(p)==h['sha256']
    p=a['experimental_panel']['points'][0]
    m[arm]={'receipt':receipt,'training':read(P/f'{arm}-training-receipt.json'),
            'guided_l1':p['initial_reconstruction']['raw_l1'],'interface':p['interface'],'recovery':p['recovery'],
            'autonomous':p['autonomous']}
s['matched_pilot']=m
plan=read(R/'plan.json')
assert identities(plan['protected'])==plan['protected']
s['final_original_hashes_verified']=True
out=Path('docs/evidence');out.mkdir(exist_ok=True)
evidence=out/'rmsnorm_followup_2026-09-09.json'
evidence.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
lines=['# RMSNorm fork follow-up — September 9, 2026','','## Technical summary','',
'Completed frozen parent/fork comparisons and a separate 32-full-core-update saturation/control pilot. '
'The longer fork combines additional training, differentiable RMSNorm and a write-logit penalty; '
'its before/after differences cannot isolate any one cause. The short pilot tests the penalty at a matched budget. '
'No broad training extension was launched. Use the per-case results below to decide the next bounded experiment.','',
'## Scope and definitions','',
'Parent step 122260; latest fork step 128196. Both frozen evaluations use OpenCL, 192px output and eight threads; '
'gradient probes use CPU. Six held-out images are measured at ages 8 and 64, with full reference or no runtime references. '
'Two familiar targets (0, 41) and two seeds (42, 137) receive 64 guided development steps and 512 autonomous steps. '
'Each macro damage case is paired with an undamaged trajectory using the same clock sequence. '
'Raw L1 is mean absolute RGB error; lower means closer reconstruction, not necessarily better autonomous dynamics. '
'Write saturation is the fraction of absolute pre-tanh logits above 2.6466525. '
'Recovery distance is the sum of micro, macro and memory RMS differences, divided by its initial value. '
'The final-interval criterion checks all saved samples from steps 384 through 512, not every intermediate step.','',
'## Held-out reconstruction','',
'| Age | Fidelity | Parent mean L1 | Fork mean L1 | Fork change |','|---:|---:|---:|---:|---:|']
for age in [8,64]:
 for f in [1,0]:
  vals=[st.mean(x['raw_l1'] for x in s['probes'] if x['arm']==arm and x['age']==age and x['reference_fidelity']==f) for arm in ['parent','fork']]
  lines.append(f'| {age} | {f} | {vals[0]:.6f} | {vals[1]:.6f} | {(vals[1]/vals[0]-1)*100:+.2f}% |')
lines+=['','Age-64 guided error improved on all six probes. The age-8 zero-reference mean worsened; the benefit is not uniform across reference conditions and ages.','',
'## Familiar target retention','', '| Target | Seed | Parent guided L1 | Fork guided L1 |','|---:|---:|---:|---:|']
for t in [0,41]:
 for seed in [42,137]:
  vals=[next(x['raw_l1'] for x in s['familiar'] if x['arm']==arm and x['target']==t and x['seed']==seed and x['phase']=='guided') for arm in ['parent','fork']]
  lines.append(f'| {t} | {seed} | {vals[0]:.6f} | {vals[1]:.6f} |')
lines+=['','These familiar targets check retention on a small panel; they do not estimate whole-corpus performance.','',
'## Write saturation and gradients','', '| Checkpoint | Head | Mean mature guided saturation | Mean tanh derivative |','|---|---|---:|---:|']
for arm in ['parent','fork']:
 for head in ['micro','macro_field']:
  rows=[x for x in s['interface'] if x['arm']==arm and x['age']==64 and x['fidelity']==1 and x['head']==head]
  lines.append(f"| {arm} | {head} | {st.mean(x['saturated_fraction'] for x in rows):.6f} | {st.mean(x['mean_tanh_derivative'] for x in rows):.6g} |")
lines+=['','The CPU gradient probes reached 0/10 normalization tensors in the parent and 10/10 (1600 scales) in the fork, with no optimizer updates. Saturation and derivative measurements above assess write responsiveness separately from state clamps.','',
'## Macro recovery','', '| Checkpoint | Mode | Damage | Median ratio 256 | Median ratio 512 | Final sampled interval below half |','|---|---|---|---:|---:|---:|']
for arm in ['parent','fork']:
 for mode in ['guided','autonomous']:
  for case in ['macro_noise','macro_patch']:
   rows=[x for x in s['recovery'] if x['arm']==arm and x['mode']==mode and x['case']==case]
   lines.append(f"| {arm} | {mode} | {case} | {st.median(x['ratio_256'] for x in rows):.6f} | {st.median(x['ratio_512'] for x in rows):.6f} | {sum(x['max_ratio_last128']<=.5 for x in rows)}/4 |")
lines+=['','Each 512-step recovery contains 128 macro updates. Ratios below one indicate contraction relative to the initial damage; threshold crossings alone do not establish sustained recovery or homeostasis. Exact per-pair endpoints and trajectories are retained in the evidence.','',
'## Matched saturation intervention','', '| Arm | Guided L1 | Mature micro saturation | Mature macro saturation | Recovery cases ending below half |','|---|---:|---:|---:|---:|']
for arm,v in m.items():
 d=next(x for x in v['interface'] if x['age']==64 and x['fidelity']==1)['trace']
 count=sum(x['trajectory'][-1]['state_distance']/x['initial_state_distance']<=.5 for x in v['recovery'])
 lines.append(f"| {arm} | {v['guided_l1']:.6f} | {d['micro']['saturated_fraction']:.6f} | {d['macro_field']['saturated_fraction']:.6f} | {count}/4 |")
lines+=['','Both arms start from identical model, world and optimizer tensor values, excluding only fork-specific identity scalars. Both enable differentiable RMSNorm and retain the optimizer and warmup position. Only the penalty differs (zero versus weight 0.0001, threshold 2.5). Each trains 256 development steps / 32 full-core updates. This is a one-target, one-seed pilot, not a causal explanation of the longer fork.','',
'## Validation and limitations','',
'All frozen checkpoint copies, original protected files and probe sources retained their hashes. All archived artifact hashes were checked. Autonomous samples record zero reference drive and absent runtime references. '
'The binary SHA-256 matches the earlier CPU/OpenCL-validated build and its 32-file Rust source snapshot; no Rust numerical code changed in this task. '
'The inherited backend mismatch was detected in an initial attempt, which was stopped and excluded before rerunning both arms on OpenCL. '
'The training log contains 289 non-increasing step transitions; only its latest contiguous 309-window invocation was used for mechanical checks. '
'Image inspection failed with the Termux filesystem sandbox error, so no visual-quality claim is made. HTML verification is structural only when Chromium is unavailable.','',
'## Decision and further questions','',
'Do not launch an unrestricted continuation based on guided reconstruction alone. The next step must address the weakest measured condition in the tables: retention if familiar errors regress, write responsiveness if saturation remains high, or autonomous macro recovery if damage does not contract. '
'A favorable one-seed penalty result warrants replication across the two-target/two-seed panel before another long block. '
'Keep RMSNorm, penalty, withdrawal, BPTT and replay changes isolated in explicit forks; this study does not establish cycles or homeostasis.','',
'## Reproduction','',
'Run `scripts/frozen_fork_followup.py --help` and `scripts/matched_saturation_pilot.py --help` for the single-use runners. Each analysis root retains the exact plan, configurations, command receipts, hashes and immutable evaluation archives. '
'Run `python scripts/summarize_frozen_fork.py analysis/rmsnorm_followup_2026-09-09_matched` to reverify the frozen evidence. '
'[Full compact evidence](evidence/rmsnorm_followup_2026-09-09.json) contains every reported comparison and archive identity.']
report=Path('docs/RMSNORM_FOLLOWUP_2026-09-09.md');report.write_text('\n'.join(lines)+'\n')
subprocess.run(['python','scripts/build_frozen_fork_report.py',str(evidence),str(report),str(R/'artifact.json')],check=True)
plugin='/data/data/com.termux/files/home/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599'
with (R/'report-delivery.json').open('w') as log:
 result=subprocess.run(['node',plugin+'/skills/build-report/scripts/deliver_portable_artifact.mjs','--input',str(R/'artifact.json'),'--output',str(R/'report.html')],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
 with report.open('a') as f:f.write('\nHTML packaging failed; the validated Markdown write-up and JSON evidence remain available. See report-delivery.json.\n')
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add',str(report),str(evidence)],check=True)
subprocess.run(['git','commit','-m','Report frozen RMSNorm recovery and matched saturation results'],check=True)
(R/'finalized.json').write_text(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'html_returncode':result.returncode})+'\n')
print('FINALIZED',flush=True)
