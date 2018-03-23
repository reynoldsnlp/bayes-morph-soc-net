"""Print progress of running models."""

from collections import defaultdict
from datetime import datetime as dt
from glob import glob
import re

logs = glob('*.log')
pickles = set([f.rstrip('.pickle') for f in glob('*.pickle')])


prog_dict = defaultdict(list)
for filename in logs:
    times = []
    progress = None
    with open(filename) as f:
        for line in f:
            match = re.search(r'Agent\s+(\d+)\s+is stepping\.\.\.', line)
            if match:
                times.append(re.match(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d+) ', line).groups())
                progress = int(match.group(1))
    if progress:
        dur = str(dt(*map(int, times[-1])) - dt(*map(int, times[0])))
        prog_dict[progress].append((filename.replace('.log', ''), dur))

warn_flag = False
star = ''
for k, v in sorted(prog_dict.items()):
    print(f'{k} ' + '=' * 50)
    for each, dur in v:
        if k == 499 and each not in pickles:
            warn_flag = True
            star = '***'
        print(f'    {star}{each}{star}\t({dur})')
        star = ''
if warn_flag:
    print('*** -- This log file seems to have completed, but there is no pickle file of the model')

