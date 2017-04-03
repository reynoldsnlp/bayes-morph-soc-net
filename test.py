"""Test out bmsn."""

import matplotlib.pyplot as plt
#import numpy as np

import bmsn


model = bmsn.MorphLearnModel(50)
for i in range(10):
    print('Model step {}.'.format(i+1))
    model.step()

aCount = 0
bCount = 0
for a in model.schedule.agents[-25:]:
    a.process_input()
    if a.morphology == 'a':
        aCount += 1
    elif a.morphology == 'b':
        bCount += 1
    else:
        print('WARN morphology is neither a nor b: {}'.format(a.morphology))

print('{}\t{}'.format(aCount, bCount))
