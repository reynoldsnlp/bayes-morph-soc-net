"""Load pickled model and extract lexeme frequency dist of last agent."""

import pickle
import sys


def key_of_highest_value(in_dict):
    """Return the key with the highest value."""
    try:
        return max(in_dict.items(), key=lambda x: x[1])[0]
    except ValueError:
        return '-'


def get_freq_dist(agent, init_dict):
    """Build dictionary from lexicon of inflection classes."""
    freq_dict = {}  # keys are tuples of endings
    if hasattr(agent, 'freq_dist'):
        return agent.freq_dist
    elif not hasattr(agent, 'in_lex_dict'):
        return 'DELETED'
    for lex, l_dict in agent.in_lex_dict.items():
        e_list = []
        for m in agent.model.seed_MSPSs:
            try:
                e_list.append(key_of_highest_value(l_dict[m]))
            except KeyError:
                # lg.warn('        {} has no value for {}!'.format(lex, m))
                e_list.append('-')
        e_list = tuple(e_list)
        try:
            freq_dict[e_list].append((lex, agent.l_dist[lex], init_dict[lex]))
        except KeyError:
            freq_dict[e_list] = [(lex, agent.l_dist[lex], init_dict[lex])]
    return freq_dict


def write_freq_dist(model, filename):
    """Write freq dists to a file with same name as pickled file."""
    init_dict = {}
    for ci, lex, freq in model.seed_lexemes():
        try:
            init_dict[lex] += freq
        except KeyError:
            init_dict[lex] = freq
    with open(filename + '_freq_dists.txt', 'w') as freq_file:
        gen_tracker = None
        for a in model.schedule.agents:
            if a.gen_id != gen_tracker:
                gen_tracker = a.gen_id
                print('=' * 79, file=freq_file)
                print('=' * 79, file=freq_file)
                print('Generation {}'.format(gen_tracker), file=freq_file)
                print('=' * 79, file=freq_file)
                print('=' * 79, file=freq_file)
            freq_dict = get_freq_dist(a, init_dict)
            print('=' * 39, file=freq_file)
            print('Gen: {},    Agent: {}'.format(a.gen_id, a.unique_id),
                  file=freq_file)
            if isinstance(freq_dict, str):
                print(freq_dict, file=freq_file)
            else:  # if freq_dict is a dict
                header = '\t'.join(['lex', 'freq', 'initFrq', 'cls_id',
                                    'exponents'])
                print(header, file=freq_file)
                for i, (e_list, class_freq_dist) in enumerate(sorted(freq_dict.items(),  # noqa
                                                                     key=lambda x: len(x[1]),  # noqa
                                                                     reverse=True)):  # noqa
                    for lexeme, freq, init_freq in class_freq_dist:
                        print(lexeme, freq, init_freq, i + 1, e_list, sep='\t',
                              file=freq_file)
    with open(filename + '_last_freq_dist.txt', 'w') as freq_file:
        a = model.schedule.agents[-1]
        freq_dict = get_freq_dist(a, init_dict)
        header = '\t'.join(['lex', 'freq', 'initFrq', 'cls_id', 'exponents'])
        print(header, file=freq_file)
        for i, (e_list, class_freq_dist) in enumerate(sorted(freq_dict.items(),
                                                             key=lambda x: len(x[1]),  # noqa
                                                             reverse=True)):
            for lexeme, freq, init_freq in class_freq_dist:
                print(lexeme, freq, init_freq, i + 1, e_list, sep='\t',
                      file=freq_file)


if __name__ == '__main__':
    filename = sys.argv[1]
    print('Input file is "{}".'.format(filename))
    filename = filename.rstrip('.pickle')
    print('Output filename is "{}".'.format(filename))

    print('Loading pickled model into memory... ({})'.format(filename))
    with open(filename + '.pickle', 'rb') as p_file:
        model = pickle.load(p_file)
    write_freq_dist(model, filename)
