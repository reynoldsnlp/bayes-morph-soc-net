"""Test out bmsn."""

import logging as lg
import pickle
import time

# import matplotlib.pyplot as plt
# import numpy as np

import bmsn

OUT_FILENAME_BASE = 'test'
ASC_TIME = time.asctime().replace(' ', '-')
GEN_SIZE = 10
GEN_COUNT = 10
H_SPACE_INC = 7
PRODUCTION_SIZE = 100000
MORPH_FILENAME = 'artificial-data/data5.txt'
START = time.time()

OUT_FILENAME = '_'.join([OUT_FILENAME_BASE,
                         ASC_TIME,
                         str(GEN_SIZE),
                         str(GEN_COUNT),
                         str(H_SPACE_INC),
                         str(PRODUCTION_SIZE)])
lg.basicConfig(filename=OUT_FILENAME + '.log', filemode='w', level=lg.DEBUG,
               format='%(asctime)s %(name)s %(levelname)s: %(message)s')
lg.getLogger().addHandler(lg.StreamHandler())

model = bmsn.MorphLearnModel(gen_size=GEN_SIZE,
                             gen_count=GEN_COUNT,
                             morph_filename=MORPH_FILENAME,
                             h_space_increment=H_SPACE_INC,
                             prod_size=PRODUCTION_SIZE)
for i in range(GEN_COUNT):
    lg.info('=' * 79)
    lg.info('Model step {}.'.format(i))
    model.step()

print('Lexeme counts: {}'.format([len(a.l_dist)
                                  for a in model.schedule.agents
                                  if hasattr(a, 'l_dist')]))

with open(OUT_FILENAME + '.pickle', 'wb') as out_file:
    pickle.dump(model, out_file)

lg.info('=' * 79)
END = time.time()
lg.info('Script took {} minutes to complete.'.format((END - START) / 60))

model_df = model.dc.get_model_vars_dataframe()
lg.info(model_df.head())
agent_df = model.dc.get_agent_vars_dataframe()
lg.info(agent_df.head())
