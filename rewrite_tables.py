"""Load pickled model and analyze data."""

import pickle
import sys

import bmsn


def get_table_dict(agent):
    """Build dictionary from lexicon of inflection classes."""
    table_dict = {}  # keys are tuples of endings
    if hasattr(agent, 'table_dict'):
        return agent.table_dict
    elif not hasattr(agent, 'in_lex_dict'):
        return 'DELETED'
    for lex, l_dict in agent.in_lex_dict.items():
        e_list = []
        for m in agent.model.seed_MSPSs:
            try:
                e_list.append(bmsn.key_of_highest_value(l_dict[m]))
            except KeyError:
                # lg.warn('        {} has no value for {}!'.format(lex, m))
                e_list.append('-')
        e_list = tuple(e_list)
        try:
            table_dict[e_list] += 1
        except KeyError:
            table_dict[e_list] = 1
    return table_dict


def write_tables(model, filename):
    """Write morphological tables to a file with same name as pickled file."""
    with open(filename + '_TABLES.txt', 'w') as table_file:
        gen_tracker = None
        for a in model.schedule.agents:
            if a.gen_id != gen_tracker:
                gen_tracker = a.gen_id
                print('=' * 79, file=table_file)
                print('=' * 79, file=table_file)
                print('Generation {}'.format(gen_tracker), file=table_file)
                print('=' * 79, file=table_file)
                print('=' * 79, file=table_file)
            table_dict = get_table_dict(a)
            print('=' * 39, file=table_file)
            print('Gen: {},    Agent: {}'.format(a.gen_id, a.unique_id),
                  file=table_file)
            if isinstance(table_dict, str):
                print(table_dict, file=table_file)
            else:
                header = '\t'.join([''] + [m for m in a.model.seed_MSPSs])
                print(header, file=table_file)
                for e_list, type_freq in sorted(table_dict.items(),
                                                key=lambda x: x[1],
                                                reverse=True):
                    print(type_freq, *e_list, sep='\t', file=table_file)


if __name__ == '__main__':
    filename = sys.argv[1]
    # if '/' in filename:
    #     filename = filename.split('/')[1:]
    print('Input file is "{}".'.format(filename))
    filename = filename.rstrip('.pickle')
    print('Output filename is "{}".'.format(filename))

    print('Loading pickled model into memory... ({})'.format(filename))
    with open(filename + '.pickle', 'rb') as p_file:
        model = pickle.load(p_file)
    write_tables(model, filename)
