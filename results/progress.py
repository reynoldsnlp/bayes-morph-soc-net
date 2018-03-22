"""Print progress of running models."""

from collections import defaultdict
from glob import glob
import re

logs = glob('*.log')
pickles = set([f.rstrip('.pickle') for f in glob('*.pickle')])


prog_dict = defaultdict(list)
for filename in logs:
    progress = None
    with open(filename) as f:
        for line in f:
            match = re.search(r'Agent\s+([0-9]+)\s+is stepping\.\.\.', line)
            if match:
                progress = int(match.group(1))
    if progress:
        prog_dict[progress].append(filename.replace('.log', ''))

warn_flag = False
star = ''
for k, v in sorted(prog_dict.items()):
    print(f'{k} ' + '=' * 50)
    for each in v:
        if k == 499 and each not in pickles:
            warn_flag = True
            star = '***'
        print(f'    {star}{each}{star}')
        star = ''
if warn_flag:
    print('*** -- This log file seems to have completed, but there is no pickle file of the model')

