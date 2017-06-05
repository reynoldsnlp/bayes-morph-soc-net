"""Test out bmsn."""

import logging as lg
import pickle
import sys
import time

import matplotlib.pyplot as plt

import bmsn

# data files to focus on: 1, 4, 7, 11
OUT_FILENAME_BASE = 'test'
ASC_TIME = time.asctime().replace(' ', '-')
GEN_SIZE = 50
GEN_COUNT = 10
H_SPACE_INC = 7
PRODUCTION_SIZE = int(sys.argv[2])
CONNECTEDNESS = 0.05
MORPH_FILENAME = sys.argv[1]
START = time.time()

OUT_FILENAME = '_'.join([OUT_FILENAME_BASE,
                         ASC_TIME,
                         str(GEN_SIZE),
                         str(GEN_COUNT),
                         str(H_SPACE_INC),
                         str(PRODUCTION_SIZE),
                         str(CONNECTEDNESS),
                         MORPH_FILENAME.split('/')[1].rstrip('.txt')])
lg.basicConfig(filename='results/' + OUT_FILENAME + '.log',
               filemode='w', level=lg.DEBUG,
               format='%(asctime)s %(name)s %(levelname)s: %(message)s')
lg.getLogger().addHandler(lg.StreamHandler())


def generationalize(in_list, N):
    """Divide flat list into generations (list of N lists)."""
    gen_size = int(len(in_list) / N)
    return [in_list[gen_size * i:gen_size * (i + 1)] for i in range(N)]


model = bmsn.MorphLearnModel(gen_size=GEN_SIZE,
                             gen_count=GEN_COUNT,
                             morph_filename=MORPH_FILENAME,
                             h_space_increment=H_SPACE_INC,
                             prod_size=PRODUCTION_SIZE,
                             connectedness=CONNECTEDNESS)
for i in range(GEN_COUNT):
    lg.info('=' * 79)
    lg.info('Model step {}.'.format(i))
    lg.info('Generation {} out of {}...'.format(i, GEN_COUNT))
    model.step()

# lg.info('{} -- Lexeme counts: {}'.format(OUT_FILENAME,
#                                          [a.data['lex_size']
#                                           for a in model.schedule.agents
#                                           if hasattr(a, 'data') and
#                                           'lex_size' in a.data]))

lg.info('building boxplots from data...')

lg.info('    lex_size...')
try:
    lex_sizes = [a.data['lex_size'] for a in model.schedule.agents]
except (AttributeError, KeyError) as error:
    lex_sizes = [bmsn.lex_size(a) for a in model.schedule.agents]
plt.boxplot(generationalize(lex_sizes, 10))
plt.savefig('results/' + OUT_FILENAME + '_lex_size.png')
plt.clf()

lg.info('    decl_entropy...')
try:
    decl_entropies = [a.data['decl_entropy'] for a in model.schedule.agents]
except (AttributeError, KeyError) as error:
    decl_entropies = [bmsn.decl_entropy(a) for a in model.schedule.agents]
plt.boxplot(generationalize(decl_entropies, 10))
plt.savefig('results/' + OUT_FILENAME + '_decl_entropy.png')
plt.clf()

lg.info('    avg_cell_entropy...')
try:
    avg_cell_entropies = [a.data['avg_cell_entropy'] for a in model.schedule.agents]  # noqa
except (AttributeError, KeyError) as error:
    avg_cell_entropies = [bmsn.avg_cell_entropy(a) for a in model.schedule.agents]  # noqa
plt.boxplot(generationalize(avg_cell_entropies, 10))
plt.savefig('results/' + OUT_FILENAME + '_avg_cell_entropy.png')
plt.clf()

lg.info('    cond_entropy...')
try:
    cond_entropies = [a.data['cond_entropy'] for a in model.schedule.agents]
except (AttributeError, KeyError) as error:
    cond_entropies = [bmsn.cond_entropy(a) for a in model.schedule.agents]
plt.boxplot(generationalize(cond_entropies, 10))
plt.savefig('results/' + OUT_FILENAME + '_cond_entropy.png')
plt.clf()

# lg.info('    bootstrap_avg and bootstrap_p...')
# try:
#     bootstrap_avgs = [a.data['bootstrap_avg'] for a in model.schedule.agents]  # noqa
#     bootstrap_ps = [a.data['bootstrap_p'] for a in model.schedule.agents]  # noqa
# except (AttributeError, KeyError) as error:
#     bootstraps = [bmsn.bootstrap(a) for a in model.schedule.agents]
#     bootstrap_avgs = [i[0] for i in bootstraps]
#     bootstrap_ps = [i[1] for i in bootstraps]
# plt.boxplot(generationalize(bootstrap_avgs, 10))
# plt.savefig('results/' + OUT_FILENAME + '_bootstrap_avg.png')
# plt.clf()
# plt.boxplot(generationalize(bootstrap_ps, 10))
# plt.savefig('results/' + OUT_FILENAME + '_bootstrap_p.png')
# plt.clf()

lg.info('=' * 79)
END = time.time()
lg.info('Script took {} minutes to complete.'.format((END - START) / 60))

# do this last just in case pickling fails
lg.info('Dumping model to pickle file...')
with open('results/' + OUT_FILENAME + '.pickle', 'wb') as out_file:
    pickle.dump(model, out_file)
