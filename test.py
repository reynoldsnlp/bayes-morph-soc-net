"""Test out bmsn."""

import pickle
import sys

# import matplotlib.pyplot as plt
# import numpy as np

import bmsn


OUT_FILENAME = 'test'
GEN_SIZE = 10
GEN_COUNT = 10
SE = sys.stderr

model = bmsn.MorphLearnModel(gen_size=GEN_SIZE, gen_count=GEN_COUNT)
for i in range(GEN_COUNT):
    print('='*79, file=SE)
    print('Model step {}.'.format(i), file=SE)
    model.step()

with open('_'.join([OUT_FILENAME,
                    str(GEN_SIZE),
                    str(GEN_COUNT),
                    '.pickle']), 'wb') as out_file:
    pickle.dump(model, out_file)

aCount = 0
bCount = 0
for a in model.schedule.agents[-GEN_SIZE:]:
    # a.process_input()
    if a.morphology == 'a':
        aCount += 1
    elif a.morphology == 'b':
        bCount += 1
    else:
        print('WARN morphology is neither a nor b: {}'.format(a.morphology))

print('='*79)
print('FINAL MORPHOLOGIES:')
print('a: {}     b: {}'.format(aCount, bCount))
