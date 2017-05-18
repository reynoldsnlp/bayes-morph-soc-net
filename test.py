"""Test out bmsn."""

import logging as lg
import pickle
import time

# import matplotlib.pyplot as plt
# import numpy as np

import bmsn

lg.basicConfig(filename='test.log', filemode='w', level=lg.DEBUG,
               format='%(asctime)s %(name)s %(levelname)s: %(message)s')
lg.getLogger().addHandler(lg.StreamHandler())

OUT_FILENAME = 'test'
GEN_SIZE = 10
GEN_COUNT = 10
MORPH_FILENAME = 'artificial-data/data5.txt'
START = time.time()

model = bmsn.MorphLearnModel(gen_size=GEN_SIZE,
                             gen_count=GEN_COUNT,
                             morph_filename=MORPH_FILENAME,
                             h_space_increment=7,
                             prod_size=10000)
for i in range(GEN_COUNT):
    lg.info('=' * 79)
    lg.info('Model step {}.'.format(i))
    model.step()

with open('_'.join([OUT_FILENAME,
                    str(GEN_SIZE),
                    str(GEN_COUNT),
                    '.pickle']), 'wb') as out_file:
    pickle.dump(model, out_file)

lg.info('=' * 79)
END = time.time()
lg.info('Script took {} minutes to complete.'.format((END - START) / 60))
