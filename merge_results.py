'''
Copyright 2025 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Merge the per-sample JSON files written by a Slurm job array (one sample per task) into a
single results file, and report which sample indices are missing so failed tasks can be
resubmitted.
'''

import argparse
import glob
import json
import re

parser = argparse.ArgumentParser(description="Merge per-sample result files from a job array")

parser.add_argument(
    '--pattern',
    type=str,
    required=True,
    help='Glob pattern for the per-sample files, e.g. "Final/garch_T500/sample_*.json"',
)

parser.add_argument(
    '--output_path',
    type=str,
    required=True,
    help='Path for the merged JSON file',
)

parser.add_argument(
    '--n_samples',
    type=int,
    help='Expected number of samples (0 to n_samples-1), used to list missing samples',
)

parser.add_argument(
    '--summary',
    action='store_true',
    help='Print iterations, final log likelihood and QHMM seconds per iteration for every sample',
)

args = parser.parse_args()

merged = None
sample_indices = set()

for path in sorted(glob.glob(args.pattern)):
    with open(path) as infile:
        data = json.load(infile)
    if merged is None:
        merged = {'metadata': data['metadata']}
    for key, value in data.items():
        if key == 'metadata':
            continue
        merged[key] = value
        sample_indices.add(int(re.search(r'sample(\d+)$', key).group(1)))

if merged is None:
    raise SystemExit(f"No files match {args.pattern}")

merged['metadata']['n_samples'] = len(sample_indices)
merged['metadata'].pop('start_sample', None)

with open(args.output_path, "w") as outfile:
    json.dump(merged, outfile, indent=4, default=str)

print(f"Merged {len(merged) - 1} entries ({len(sample_indices)} samples) into {args.output_path}")
if args.n_samples:
    missing = sorted(set(range(args.n_samples)) - sample_indices)
    print(f"Missing samples: {missing if missing else 'none'}")

if args.summary:
    # An empty training curve or 0 iterations means that model's fit failed
    models = ['qhmm', 'npc', 'pc', 'garch']
    print(f"{'sample':<32}" + ''.join(f"{name + ' nit':>10}{name + ' ll':>12}" for name in models) + f"{'qhmm s/it':>11}")
    for key, value in merged.items():
        if key == 'metadata':
            continue
        row = f"{key:<32}"
        for name in models:
            curve = value.get(f'training_curve_{name}', [])
            final = f"{-curve[-1]:.2f}" if curve else 'FAILED'
            row += f"{value.get(f'nit_{name}', 0):>10}{final:>12}"
        nit_q = value.get('nit_qhmm', 0)
        row += f"{value.get('training_time_qhmm', 0) / nit_q:>11.1f}" if nit_q else f"{'-':>11}"
        print(row)
