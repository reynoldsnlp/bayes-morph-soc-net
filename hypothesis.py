"""Hypothesis space for Bayesian learning model."""

from collections import OrderedDict  # TODO(RJR) costly, is it necessary?

from scipy.stats import entropy as KL_divergence

from morphology import Morphology


class Hypothesis:
    """Hypothesis space for Bayesian learning."""

    def __init__(self, morphology):
        """Build hypthesis space given a morphology object."""
        self.h_spaces = OrderedDict()  # dict of dicts
        for h_set in morphology.mnb:
            print('h_set:\t{}'.format(h_set))
            list_of_endings = [i for i in morphology.msps_dict[h_set[0]]]
            print('list_of_endings\t{}'.format(list_of_endings))
            h_space = self.generate_hyp_space(list_of_endings, 5)
            # print('h_space\t{}'.format(h_space))
            self.h_spaces[h_set] = h_space

    def stars_and_bars(self, n, k, the_list=[]):
        """Distribute n probability tokens among k endings.

        Generator implementation of the stars-and-bars algorithm.

        Arguments:
        n   --  number of probability tokens to divide among bins (stars)
        k   --  number of endings/bins
        """
        if n == 0:
            yield the_list + [0]*k
        elif k == 1:
            yield the_list + [n]
        else:
            for i in range(n+1):
                yield from self.stars_and_bars(n-i, k-1, the_list+[i])

    def generate_hyp_space(self, list_of_items, increment_divisor=None):
        """Generate list of OrderedDicts filling the hypothesis space.

        Each OrderedDict is of the form ...
        {i1: 0.0, i2: 0.1, i3: 0.0, ...}
        ... where .values() sums to 1.

        Arguments:
        list_of_items     -- items that receive prior weights
        increment_divisor -- Increment by 1/increment_divisor. For example,
                             4 yields (0.0, 0.25, 0.5, 0.75, 1.0).
        """
        _LEN = len(list_of_items)
        # TODO(RJR) compute increment_divisor dynamically? len(list_of_items)
        if increment_divisor is None:  # TODO(RJR) or increment_divisor < _LEN:
            increment_divisor = _LEN
            # print('WARN: increment_divisor smaller than len(list_of_items).')
        # TODO(RJR) Q: what if only one ending is possible? (as in InstPl)
        h_space = []
        for perm in self.stars_and_bars(increment_divisor, _LEN):
            perm = [s/increment_divisor for s in perm]
            h_space.append(OrderedDict([(list_of_items[i], perm[i])
                                        for i in range(_LEN)]))
        return h_space


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile
    from os.path import join

    DATA_DIR = 'language-data'

    filenames = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    morphologies = []
    for filename in filenames:
        print('Importing {}...'.format(filename))
        with open(DATA_DIR+'/'+filename) as morph_file:
            morphologies.append(Morphology(morph_file))

    rus_morph = morphologies[6]
    rus_h = Hypothesis(rus_morph)
    print(len(rus_h.h_spaces))
    print(sum([len(i) for i in rus_h.h_spaces]))
