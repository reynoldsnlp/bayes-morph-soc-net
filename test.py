"""Test out bmsn.

$ python3 test.py <morph_table> <production_size> <prior_weight> <lexeme_dist_shape>
"""


import datetime as dt
import logging as lg
import pickle
import sys
import time

import matplotlib.pyplot as plt

import bmsn
import rewrite_tables as rt


# TODO(RJR) implement argparse
OUT_FILENAME_BASE = 'test'
TIMESTAMP = dt.datetime.now().isoformat().replace(':', '-').replace('.', '-')
GEN_SIZE = 50
GEN_COUNT = 10
H_SPACE_INC = 7
RAND_TF = True
try:
    MORPH_FILENAME = sys.argv[1]
except IndexError:
    MORPH_FILENAME = 'unknown'
try:
    PRODUCTION_SIZE = int(sys.argv[2])
except IndexError:
    PRODUCTION_SIZE = 100000
CONNECTEDNESS = 0.05
try:
    PRIOR_WEIGHT = sys.argv[3]
    if PRIOR_WEIGHT in ['', 'None', 'none']:
        PRIOR_WEIGHT = None
except IndexError:
    PRIOR_WEIGHT = None
try:
    LEXEME_DIST_SHAPE = sys.argv[4]
except IndexError:
    LEXEME_DIST_SHAPE = 'flat'
START = time.time()

if RAND_TF:
    rand = 'rand'
else:
    rand = ''
OUT_FILENAME = '_'.join([OUT_FILENAME_BASE,
                         TIMESTAMP, '', '',  # add underscores
                         str(GEN_SIZE),
                         str(GEN_COUNT),
                         str(H_SPACE_INC),
                         str(PRODUCTION_SIZE),
                         str(CONNECTEDNESS),
                         rand,
                         MORPH_FILENAME.split('/')[1].rstrip('.txt')])


def generationalize(in_list, N):
    """Divide flat list into generations (list of N lists)."""
    gen_size = int(len(in_list) / N)
    return [in_list[gen_size * i:gen_size * (i + 1)] for i in range(N)]


def get_data(agent, function, data_key):
    """Retrieve cached datapoint or compute new datapoint."""
    try:
        return agent.data[data_key]
    except KeyError:
        return function(agent)


def draw_boxplots(model, out_filename=OUT_FILENAME, bootstrapping=False):
    """Draw boxplots of agents' attributes across generations."""
    lg.info('Running draw_boxplots...')
    gen_count = model.gen_count

    lg.info('    lex_size...')
    try:
        lex_sizes = [get_data(a, bmsn.lex_size, 'lex_size')
                     for a in model.schedule.agents]
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        lex_sizes = [bmsn.lex_size(a) for a in model.schedule.agents]
    ymax = max(lex_sizes)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
    plt.boxplot(generationalize(lex_sizes, gen_count))
    plt.savefig('results/' + out_filename + '_lex_size.png')
    plt.clf()

    lg.info('    class_count...')
    try:
        class_counts = [get_data(a, bmsn.class_counter, 'class_count')
                        for a in model.schedule.agents]
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        class_counts = [bmsn.class_counter(a) for a in model.schedule.agents]
    ymax = max(class_counts)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
    plt.boxplot(generationalize(class_counts, gen_count))
    plt.savefig('results/' + out_filename + '_class_count.png')
    plt.clf()

    lg.info('    exp_count...')
    try:
        exp_counts = [get_data(a, bmsn.exp_counter, 'exp_count')
                      for a in model.schedule.agents]
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        exp_counts = [bmsn.exp_counter(a) for a in model.schedule.agents]
    ymax = max(exp_counts)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
    plt.boxplot(generationalize(exp_counts, gen_count))
    plt.savefig('results/' + out_filename + '_exp_count.png')
    plt.clf()

    lg.info('    decl_entropy...')
    try:
        decl_entropies = [get_data(a, bmsn.decl_entropy, 'decl_entropy')
                          for a in model.schedule.agents]
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        decl_entropies = [bmsn.decl_entropy(a) for a in model.schedule.agents]
    ymax = max(decl_entropies)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
        lg.warn('maximum value over 2.0!')
    plt.boxplot(generationalize(decl_entropies, gen_count))
    plt.savefig('results/' + out_filename + '_decl_entropy.png')
    plt.clf()

    lg.info('    avg_cell_entropy...')
    try:
        avg_cell_entropies = [get_data(a, bmsn.avg_cell_entropy, 'avg_cell_entropy')  # noqa
                              for a in model.schedule.agents]  # noqa
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        avg_cell_entropies = [bmsn.avg_cell_entropy(a) for a in model.schedule.agents]  # noqa
    ymax = max(avg_cell_entropies)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
        lg.warn('maximum value over 2.0!')
    plt.boxplot(generationalize(avg_cell_entropies, gen_count))
    plt.savefig('results/' + out_filename + '_avg_cell_entropy.png')
    plt.clf()

    lg.info('    cond_entropy...')
    try:
        cond_entropies = [get_data(a, bmsn.cond_entropy, 'cond_entropy')
                          for a in model.schedule.agents]
    except (AttributeError, KeyError) as error:
        lg.info('        ...not in model. Computing from scratch...')
        cond_entropies = [bmsn.cond_entropy(a) for a in model.schedule.agents]
    ymax = max(cond_entropies)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
        lg.warn('maximum value over 2.0!')
    plt.boxplot(generationalize(cond_entropies, gen_count))
    plt.savefig('results/' + out_filename + '_cond_entropy.png')
    plt.clf()

    lg.info('    mut_info...')
    mut_infos = [i - j for i, j in zip(avg_cell_entropies, cond_entropies)]
    ymax = max(mut_infos)
    if ymax <= 2.0:
        plt.ylim((-0.1, 2.1))
    else:
        plt.ylim((-0.1, ymax + 0.1))
        lg.warn('maximum value over 2.0!')
    plt.boxplot(generationalize(mut_infos, gen_count))
    plt.savefig('results/' + out_filename + '_mut_info.png')
    plt.clf()

    if bootstrapping:
        lg.info('    bootstrap_avg and bootstrap_p...')
        # TODO(RJR) update to use new function like get_data
        try:
            bootstrap_avgs = [a.data['bootstrap_avg'] for a in model.schedule.agents]  # noqa
            bootstrap_ps = [a.data['bootstrap_p'] for a in model.schedule.agents]  # noqa
        except (AttributeError, KeyError) as error:
            lg.info('        ...not in model. Computing from scratch...')
            bootstraps = [bmsn.bootstrap(a) for a in model.schedule.agents]
            bootstrap_avgs = [i[0] for i in bootstraps]
            bootstrap_ps = [i[1] for i in bootstraps]
        ymax = max(bootstrap_avgs)
        if ymax <= 2.0:
            plt.ylim((-0.1, 2.1))
        else:
            plt.ylim((-0.1, ymax + 0.1))
            lg.warn('maximum value over 2.0!')
        plt.boxplot(generationalize(bootstrap_avgs, gen_count))
        plt.savefig('results/' + out_filename + '_bootstrap_avg.png')
        plt.clf()
        ymax = max(bootstrap_ps)
        if ymax <= 2.0:
            plt.ylim((-0.1, 2.1))
        else:
            plt.ylim((-0.1, ymax + 0.1))
            lg.warn('maximum value over 2.0!')
        plt.boxplot(generationalize(bootstrap_ps, gen_count))
        plt.savefig('results/' + out_filename + '_bootstrap_p.png')
        plt.clf()


if __name__ == '__main__':
    lg.basicConfig(filename='results/' + OUT_FILENAME + '.log',
                   filemode='w', level=lg.DEBUG,
                   format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    # lg.getLogger().addHandler(lg.StreamHandler())  # echoes log to stderr
    model = bmsn.MorphLearnModel(out_filename=OUT_FILENAME,
                                 gen_size=GEN_SIZE,
                                 gen_count=GEN_COUNT,
                                 morph_filename=MORPH_FILENAME,
                                 h_space_increment=H_SPACE_INC,
                                 prod_size=PRODUCTION_SIZE,
                                 connectedness=CONNECTEDNESS,
                                 rand_tf=RAND_TF,
                                 lexeme_dist_shape=LEXEME_DIST_SHAPE,
                                 prior_weight=PRIOR_WEIGHT)
    for i in range(GEN_COUNT):
        lg.info('=' * 79)
        lg.info('Model step {}.'.format(i))
        lg.info('Generation {} out of {}...'.format(i, GEN_COUNT))
        model.step()

    bmsn.metalog(model)
    draw_boxplots(model)
    rt.write_tables(model, 'results/' + OUT_FILENAME)

    lg.info('=' * 79)
    END = time.time()
    lg.info('Script took {} minutes to complete.'.format((END - START) / 60))

    # do this last just in case pickling fails
    lg.info('Dumping model to pickle file...')
    with open('results/' + OUT_FILENAME + '.pickle', 'wb') as out_file:
        pickle.dump(model, out_file)
