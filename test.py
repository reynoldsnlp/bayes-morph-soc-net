"""Test out bmsn."""

import sys

# import matplotlib.pyplot as plt
# import numpy as np

import bmsn


GEN_SIZE = 10
GEN_COUNT = 10
SE = sys.stderr

model = bmsn.MorphLearnModel(gen_size=GEN_SIZE, gen_count=GEN_COUNT)
for i in range(10):
    print('='*79, file=SE)
    print('Model step {}.'.format(i+1), file=SE)
    model.step()

aCount = 0
bCount = 0
for a in model.schedule.agents[-GEN_SIZE:]:
    a.process_input()
    if a.morphology == 'a':
        aCount += 1
    elif a.morphology == 'b':
        bCount += 1
    else:
        print('WARN morphology is neither a nor b: {}'.format(a.morphology))

print('{}\t{}'.format(aCount, bCount))
