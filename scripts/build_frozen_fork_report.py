#!/usr/bin/env python3
"""Package the reviewed Git write-up and frozen comparison data for the report reader."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('writeup', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    assert data['complete'], 'Report requires all planned frozen evaluations'
    narrative = args.writeup.read_text()
    title = narrative.splitlines()[0].removeprefix('# ')
    blocks = []
    for index, section in enumerate(re.split(r'(?=^## )', narrative, flags=re.MULTILINE)):
        blocks.append(dict(id=f'text-{index}', type='markdown', body=section.strip(), sourceId='writeup'))
    source = dict(id='measurements', label='Verified frozen checkpoint comparison',
                  path='docs/evidence/rmsnorm_followup_2026-09-09.json')
    sources = [source, dict(id='writeup', label='Reviewed experiment write-up',
                           path='docs/RMSNORM_FOLLOWUP_2026-09-09.md')]
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.create_function('readfile', 1, lambda path: Path(path).read_text())
    sql_path = str(args.summary).replace("'", "''")
    datasets = {}
    charts = []
    tables = []

    def chart(key, title, rows, x, y, kind='bar', x_type='nominal', y_label=None):
        if key == 'probe-guided':
            query = f"SELECT json_extract(value, '$.arm') AS arm, replace(replace(json_extract(value, '$.name'), '.png', ''), '_seed1045551771', '') AS probe, json_extract(value, '$.raw_l1') AS raw_l1 FROM json_each(readfile('{sql_path}'), '$.probes') WHERE json_extract(value, '$.age') = 64 AND json_extract(value, '$.reference_fidelity') = 1"
        else:
            mode, case = key.split('-', 1)
            assert mode in ['guided', 'autonomous'] and case in ['macro_noise', 'macro_patch']
            query = f"SELECT json_extract(value, '$.arm') AS arm, json_extract(value, '$.offset') AS offset, avg(json_extract(value, '$.ratio')) AS ratio, count(*) AS pairs FROM json_each(readfile('{sql_path}'), '$.trajectories') WHERE json_extract(value, '$.mode') = '{mode}' AND json_extract(value, '$.case') = '{case}' GROUP BY arm, offset ORDER BY arm, offset"
        reviewed = [dict(row) for row in db.execute(query)]
        assert len(reviewed) == len(rows)
        datasets[key] = reviewed
        chart_source = dict(source, query=dict(engine='SQLite JSON1 with readfile supplied by Python', language='sql',
            sql=query, description='Query executed against the verified frozen comparison JSON; readfile reads the named local evidence file.',
            tables_used=[str(args.summary)]))
        charts.append(dict(id=key, title=title, type=kind, dataset=key, source=chart_source,
            encodings=dict(x=dict(field=x, type=x_type, label=x.replace('_', ' ').title()),
                           y=dict(field=y, type='quantitative', label=y_label or y.replace('_', ' ').title()),
                           color=dict(field='arm', type='nominal', label='Checkpoint'))))
        return dict(id=f'block-{key}', type='chart', chartId=key, layout='full')

    def table(key, title, rows, columns):
        datasets[key] = rows
        tables.append(dict(id=key, title=title, dataset=key, sourceId='measurements',
            defaultSort=dict(field=columns[0], direction='asc'),
            columns=[dict(field=k, label=k.replace('_',' ').title(),
                          type='number' if isinstance(rows[0].get(k), (int,float)) else 'text') for k in columns]))
        return dict(id=f'block-{key}', type='table', tableId=key)

    # The write-up is the canonical narrative. Add evidence after its matching section.
    for block in list(blocks):
        body = block['body']
        inserts = []
        if body.startswith('## Held-out'):
            rows=[dict(arm=x['arm'],probe=x['name'].removesuffix('.png').replace('_seed1045551771',''),raw_l1=x['raw_l1'])
                  for x in data['probes'] if x['age']==64 and x['reference_fidelity']==1]
            inserts.append(chart('probe-guided', 'Guided fractal error at age 64', rows, 'probe', 'raw_l1',y_label='Raw RGB L1'))
        elif body.startswith('## Familiar'):
            rows=[{k:x[k] for k in ['arm','target','seed','raw_l1']} for x in data['familiar'] if x['phase']=='guided']
            inserts.append(table('familiar', 'Familiar target error after guided burn-in', rows,['target','seed','arm','raw_l1']))
        elif body.startswith('## Write'):
            rows=[{k:x[k] for k in ['arm','target','seed','head','saturated_fraction','mean_tanh_derivative','absolute_logit_p95']}
                  for x in data['interface'] if x['age']==64 and x['fidelity']==1]
            inserts.append(table('saturation', 'Mature guided writes', rows,
                                  ['target','seed','arm','head','saturated_fraction','mean_tanh_derivative','absolute_logit_p95']))
        elif body.startswith('## Macro'):
            for mode in ['guided','autonomous']:
                for case in ['macro_noise','macro_patch']:
                    groups=defaultdict(list)
                    for x in data['trajectories']:
                        if x['mode']==mode and x['case']==case:
                            groups[(x['arm'],x['offset'])].append(x['ratio'])
                    rows=[dict(arm=arm,offset=offset,ratio=statistics.mean(v),pairs=len(v))
                          for (arm,offset),v in sorted(groups.items())]
                    key=mode+'-'+case
                    inserts.append(dict(id=key+'-note',type='markdown',
                        body=f'### {mode.title()} {case.replace("_", " ")}\n\n'
                             'The curves show mean damaged/control state distance divided by its initial value, '
                             'across four target/seed pairs. Lower values mean closer trajectories; '
                             'the per-pair endpoints remain necessary to assess variability.',sourceId='measurements'))
                    inserts.append(chart(key, f'{mode.title()} {case.replace("_", " ")} recovery', rows,
                                         'offset','ratio','line','quantitative','Distance / initial distance'))
            rows=[{k:x[k] for k in ['arm','target','seed','mode','case','ratio_256','ratio_512','max_ratio_last128']}
                  for x in data['recovery']]
            inserts.append(table('recovery-endpoints','Recovery by target and seed',rows,
                                 ['target','seed','arm','mode','case','ratio_256','ratio_512','max_ratio_last128']))
        if inserts:
            at=blocks.index(block)+1
            blocks[at:at]=inserts
    now=datetime.now(timezone.utc).isoformat()
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,
        generatedAt=now,blocks=blocks,charts=charts,tables=tables,sources=sources),
        snapshot=dict(version=1,status='ready',generatedAt=now,datasets=datasets),sources=sources)
    args.output.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
