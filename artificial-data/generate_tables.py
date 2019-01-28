"""Generate artificial input morphological tables."""

from collections import Counter
from pathlib import Path
from pprint import pprint
from random import choice
from random import choices
from statistics import mean
from string import ascii_lowercase
from string import ascii_uppercase
# import sys

from matplotlib import pyplot as plt
from networkx import Graph

from morph_nx import MorphGraph


Path('autogen').mkdir(exist_ok=True)


class MorphTable():
    """Tab-separated morphological table of the following format:

    typeFreq	A	B	C	D	E	F
    NA	a	x	jj	uu	ww	ttt
    NA	a	m	jj	vv	ww	iii
    NA	b	m	y	vv	xx	iii
    NA	b	n	y	kk	xx	jjj
    NA	c	n	z	kk	yy	jjj
    NA	c	o	z	ll	yy	kkk
    ...
    """

    def __init__(self, source=None, e_min=2, e_max=24, num_cells=6,
                 num_classes=24):
        """Generate a morphological table."""
        self.source = source
        self.e_min = e_min
        self.e_max = e_max
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.AMs = self.gen_AMs()
        if self.source is None:
            self.AM_dict = {i: [next(self.AMs)
                                for _ in range(choice(range(e_min, e_max)))]
                            for i in range(num_cells)}
            self.gen_random_table()
            self.get_metrics()
        elif isinstance(self.source, Graph):  # networkx Graph
            self.get_table_from_graph()
            self.get_metrics()
            try:
                assert self.mean_degree == self.source.avg_deg()
                assert self.mean_edge_weight == self.source.avg_wght()
            except AssertionError as e:
                print(f'MorphGraph metrics:\t{self.source.avg_deg():.4f}\t'
                      f'{self.source.avg_wght():.4f}')
                print(f'MorphTable metrics:\t{self.mean_degree:.4f}\t'
                      f'{self.mean_edge_weight:.4f}')
                # raise e
        elif isinstance(self.source, str):  # filename
            self.get_table_from_file()
            self.get_metrics()
        else:
            print(type(self.source))
            raise NotImplementedError

    def __repr__(self):
        repr = ['typeFreq\t' + '\t'.join(ascii_uppercase[:self.num_cells])]
        for r in self.tbl:
            repr.append('NA\t' + '\t'.join(r))
        return '\n'.join(repr)

    def gen_random_table(self):
        """Generate table randomly."""
        self.tbl = set()
        while len(self.tbl) < self.num_classes:
            self.tbl.add(tuple(choice(self.AM_dict[i])
                               for i in range(self.num_cells)))
        self.tbl = list(self.tbl)

    def get_table_from_graph(self):
        """Generate table structure from MorphGraph (custom networkx Graph)."""
        self.tbl = [[None] * self.num_cells for _ in range(self.num_classes)]
        g = self.source
        # Place exponents in the table
        for u, v, d in g.edges(data=True):
            shared_MSPSs = d['weight']
            for msps in shared_MSPSs:
                rows = {u, v}  # all rows that share this exp
                just_saw_a_new_one = True
                while just_saw_a_new_one:
                    just_saw_a_new_one = False
                    for U, V, D in g.edges(data=True):
                        if U in rows and msps in D['weight'] and V not in rows:
                            rows.add(V)
                            just_saw_a_new_one = True
                        if V in rows and msps in D['weight'] and U not in rows:
                            rows.add(U)
                            just_saw_a_new_one = True
                existing_exps = set(self.tbl[r][msps] for r in rows) - {None}
                lee = len(existing_exps)
                if lee == 0:
                    exponent = '_' + next(self.AMs)
                if lee == 1:
                    continue
                elif lee > 1:
                    pprint(self.tbl)
                    print(rows, msps)
                    raise AttributeError('Whaaaaa?!')
                for r in rows:
                    self.tbl[r][msps] = exponent
        self.tbl = [[e or next(self.AMs) for e in row] for row in self.tbl]

    def get_table_from_file(self):
        """Import table structure from tab-separated file."""
        with Path(self.source).open() as f:
            self.tbl = []
            for line in f.readlines()[1:]:
                self.tbl.append(tuple(line.strip().split('\t')[1:]))
                assert len(set([len(r) for r in self.tbl])) == 1
        self.num_cells = len(self.tbl[0])
        self.num_classes = len(self.tbl)

    def get_col(self, index):
        """Return column of `self.tbl`."""
        return [row[index] for row in self.tbl]

    def get_matching_rows(self, msps, exponent):
        return [i for i, row in enumerate(self.tbl)
                if self.tbl[i][msps] == exponent]

    @staticmethod
    def gen_AMs():
        """Generate list of unique 'allomorphs'."""
        gem = 0  # duplication factor
        while True:
            gem += 1
            for c in ascii_lowercase:
                yield c * gem

    def get_metrics(self):
        """Calculate mean degree and mean edge weight.

        Mean degree - On average, how many nodes is each node connected to
        Mean edge weight - On average, how many classes share each exponent
        """
        degs = []
        weights = []
        for row in range(self.num_classes):
            cell_counter = Counter()
            for other_row in range(self.num_classes):
                if other_row != row:
                    for msps in range(self.num_cells):
                        if self.tbl[other_row][msps] == self.tbl[row][msps]:
                            cell_counter.update([other_row])
            degs.append(len(cell_counter))
            weights.extend(cell_counter.values())
        md = mean(degs)
        self.mean_degree = md
        self.norm_mean_degree = md / (self.num_classes - 1)
        mew = mean(weights)
        self.mean_edge_weight = mew
        self.norm_mean_edge_weight = mew / (self.num_cells - 1)

    def mutate(self, MSPSs=1, cells=1, guaranteed=False):
        """Mutate table by changing x exponents in each MSPS.

        x          -- How many cells to change per MSPS
        prob       -- Probability that any given MSPS will actually be changed
        guaranteed -- Guarantee that exponent is not replaced by itself, unless
                      there is only one exponent for that MSPS
        """
        for msps in choices(range(self.num_cells), k=MSPSs):
            already_changed = set()
            for _ in range(cells):
                pool = [row[msps] for row in self.tbl]
                victim_row = choice(range(self.num_classes))
                while ((msps, victim_row) in already_changed
                       and len(already_changed) < len(self.num_classes)):
                    victim_row = choice(range(self.num_classes))
                new_exp = choice(pool)
                if guaranteed and len(set(pool)) > 1:
                    while self.tbl[victim_row][msps] == new_exp:
                        new_exp = choice(pool)
                new_row = list(self.tbl[victim_row])
                new_row[msps] = new_exp
                self.tbl[victim_row] = tuple(new_row)
                already_changed.add((msps, victim_row))
        self.get_metrics()


if __name__ == '__main__':
    NORM_LIMS = (-0.1, 1.1)
    from_nx = []
    for i in range(1, 24):
        for j in range(1, 6):
            print(i, j)
            g = MorphGraph(24, i, j)
            print(g.avg_deg(), g.avg_wght())
            from_nx.append(MorphTable(g))
    nx_mean_degs = [mt.norm_mean_degree for mt in from_nx]
    nx_mean_weights = [mt.norm_mean_edge_weight for mt in from_nx]
    plt.scatter(nx_mean_degs, nx_mean_weights, marker='o', color='r')
    plt.title('Random generation of tables from networkx Graphs')
    plt.xlabel('Mean degree')
    plt.ylabel('Mean edge weight')
    plt.xlim(NORM_LIMS)
    plt.ylim(NORM_LIMS)
    plt.show()

    jeffs = [MorphTable(fname) for fname in Path('.').glob('data_6.*')]
    orig_mean_degs = [mt.norm_mean_degree for mt in jeffs]
    orig_mean_weights = [mt.norm_mean_edge_weight for mt in jeffs]
    plt.scatter(orig_mean_degs, orig_mean_weights, marker='o', color='r')
    mean_degs = []
    mean_weights = []
    for mt in jeffs:
        for i in range(100):
            mt.mutate()
            mean_degs.append(mt.norm_mean_degree)
            mean_weights.append(mt.norm_mean_edge_weight)

    # print(f'writing file {i}...', file=sys.stderr)
    # with Path(f'autogen/{str(i).zfill(4)}.txt').open('w') as f:
    #     print(mt, file=f)

    plt.scatter(mean_degs, mean_weights, marker='.')
    plt.title('Incremental random mutation of 6.x tables')
    plt.xlabel('Normalized mean degree')
    plt.ylabel('Normalized mean edge weight')
    plt.xlim(NORM_LIMS)
    plt.ylim(NORM_LIMS)
    # axes = plt.gca()
    # xlim = axes.get_xlim()
    # ylim = axes.get_ylim()
    plt.show()

    mean_degs = []
    mean_weights = []
    for i in range(1000):
        mt = MorphTable()
        mean_degs.append(mt.norm_mean_degree)
        mean_weights.append(mt.norm_mean_edge_weight)
    plt.scatter(mean_degs, mean_weights, marker='.')
    plt.title('Random generation of tables')
    plt.xlabel('Mean degree')
    plt.ylabel('Mean edge weight')
    plt.xlim(NORM_LIMS)
    plt.ylim(NORM_LIMS)
    plt.show()

    langs = [(MorphTable(f), f.stem)
             for f in Path('../language-data').glob('*.txt')]
    lang_mean_degs = [mt.norm_mean_degree for mt, name in langs]
    lang_mean_weights = [mt.norm_mean_edge_weight for mt, name in langs]
    lang_names = [name for mt, name in langs]
    fix, ax = plt.subplots()
    ax.scatter(lang_mean_degs, lang_mean_weights, marker='o', color='b')
    for x, y, txt in zip(lang_mean_degs, lang_mean_weights, lang_names):
        ax.annotate(txt, (x, y))
    mean_degs = []
    mean_weights = []
    for mt, name in langs:
        for i in range(100):
            mt.mutate()
            mean_degs.append(mt.norm_mean_degree)
            mean_weights.append(mt.norm_mean_edge_weight)
    plt.scatter(mean_degs, mean_weights, marker='.')
    plt.title('Incremental random mutation of natural language tables')
    plt.xlabel('Normalized mean degree')
    plt.ylabel('Normalized mean edge weight')
    plt.xlim(NORM_LIMS)
    plt.ylim(NORM_LIMS)
    plt.show()
