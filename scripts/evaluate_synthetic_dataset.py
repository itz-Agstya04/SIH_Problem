import csv
from pathlib import Path

from app.ml import evaluate_cross_validated

out = Path(__file__).resolve().parent.parent / 'data' / 'synthetic_sif_eval.csv'
out.parent.mkdir(exist_ok=True)

positive = [
    'worker entered energized panel while loto was not verified and exposure was present',
    'technician near live panel without isolation and barrier failure during maintenance',
    'lifting operation with suspended load, line of fire, inadequate guarding and worker exposure',
    'hot work with sparks near flammable material and no permit to work',
    'reversing vehicle struck worker in line of fire after spotter was absent',
    'confined space entry with hazardous atmosphere and gas testing not performed',
    'fall from height while working on scaffold and guardrail failed',
    'electrical shock from wrong breaker isolated and verification not performed',
    'pressure release from hydraulic line near technician with barrier bypassed',
    'crane suspended load over worker with missing barricade and exposed area',
]

negative = [
    'routine housekeeping inspection completed with no hazard exposure',
    'office administrative review and paperwork update completed safely',
    'general plant walkdown identified no unsafe act or electrical energy',
    'standard toolbox talk delivered and work completed without incident',
    'routine equipment check with control room observation and no exposure',
    'warehouse inventory task with no energized equipment and no line of fire',
    'daily inspection of office area with no hazardous work',
    'scheduled maintenance planning meeting with no field work',
    'inspection of storage racks with no moving equipment or barriers failed',
    'document control review and training record update completed without exposure',
]

rows = []
for i in range(40):
    rows.append({
        'report_id': f'POS-{i:03d}',
        'site': 'Site A',
        'activity': 'Maintenance',
        'report_type': 'Near miss',
        'raw_description': positive[i % len(positive)] + f' sample {i}',
        'human_label': 'sif_potential',
    })

for i in range(40):
    rows.append({
        'report_id': f'NEG-{i:03d}',
        'site': 'Site B',
        'activity': 'Administration',
        'report_type': 'Observation',
        'raw_description': negative[i % len(negative)] + f' sample {i}',
        'human_label': 'not_sif_potential',
    })

with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['report_id','site','activity','report_type','raw_description','human_label'])
    writer.writeheader()
    writer.writerows(rows)

texts = [row['raw_description'] for row in rows]
labels = [1 if row['human_label'] == 'sif_potential' else 0 for row in rows]
metrics = evaluate_cross_validated(texts, labels)
print('rows=', len(rows))
print('balance=', {int(label): labels.count(label) for label in sorted(set(labels))})
print(metrics)
