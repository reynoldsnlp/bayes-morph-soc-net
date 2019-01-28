from itertools import combinations as combos
from statistics import mean
from statistics import StatisticsError
from random import choice
from random import random
from random import sample
import sys
from timeit import default_timer as timer

import networkx as nx


class MorphGraph(nx.Graph):
    def __init__(self, num_nodes, target_deg, target_wght, max_wght=5,
                 debug=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.target_deg = target_deg
        self.target_wght = target_wght
        self.max_wght = max_wght
        self.MSPSs = list(range(self.max_wght + 1))
        self.MSPS_set = set(self.MSPSs)
        self.add_nodes_from(range(self.num_nodes))
        self.deg_in_range = False
        self.wght_in_range = False
        iter_count = 0
        while not (self.deg_in_range and self.wght_in_range):
            iter_count += 1
            if iter_count > 5000:
                break
            deg_diff = self.deg_diff()
            wght_diff = self.wght_diff()
            # if abs(deg_diff) / target_deg >= abs(wght_diff) / target_wght:
            if random() > 0.5:  # randomly decide to change edge vs wght
                if deg_diff < 0:
                    if debug:
                        print('add random edge', deg_diff, file=sys.stderr)
                    u, v = sample(self.nodes(), 2)
                    while v in self[u]:
                        u, v = sample(self.nodes(), 2)
                    msps = choice(self.MSPSs)
                    self.add_edge(u, v, weight=set([msps]))
                    self.add_propagate(u, v, msps)
                else:
                    if debug:
                        print('remove random edge (all MSPSs)', deg_diff,
                              file=sys.stderr)
                    buhbye = choice(list(self.edges()))
                    self.remove_edge(*buhbye)
            else:
                if wght_diff < 0:
                    if debug:
                        print('add random msps to already-existing edge',
                              wght_diff, file=sys.stderr)
                    try:
                        u, v = choice([e for e in self.edges()
                                       if len(self[e[0]][e[1]]['weight'])
                                       < self.max_wght])
                    except IndexError:
                        continue
                    msps = choice(list(self.MSPS_set - self[u][v]['weight']))
                    self[u][v]['weight'].add(msps)
                    self.add_propagate(u, v, msps)
                else:
                    if debug:
                        print('remove random msps from existing edge',
                              wght_diff, file=sys.stderr)
                    try:
                        u, v = choice([e for e in self.edges() if
                                       len(self[e[0]][e[1]]['weight']) > 1])
                    except IndexError:
                        print('\tWARN: No edges exist with more than 1 MSPS',
                              file=sys.stderr)
                        continue
                    msps = self[u][v]['weight'].pop()
                    self.drop_propagate(u, v, msps)
            self.deg_in_range = abs(self.deg_diff()) < 1
            self.wght_in_range = abs(self.wght_diff()) < 1
        if self.deg_in_range and self.wght_in_range:
            self.success = True
        else:
            self.success = False

    def add_propagate(self, u, v, msps):
        nodes = {u, v}
        just_saw_a_new_one = True
        while just_saw_a_new_one:
            just_saw_a_new_one = False
            for U, V, D in self.edges(data=True):
                if U in nodes and msps in D['weight'] and V not in nodes:
                    nodes.add(V)
                    just_saw_a_new_one = True
                if V in nodes and msps in D['weight'] and U not in nodes:
                    nodes.add(U)
                    just_saw_a_new_one = True
        for u, v in combos(nodes, 2):
            try:
                self[u][v]['weight'].add(msps)
            except KeyError:
                self.add_edge(u, v, weight={msps})

    def drop_propagate(self, u, v, msps):
        # randomly choose which node is losing all connections on this msps
        drop_node = choice([u, v])
        for other_node in self[drop_node]:
            try:
                self[drop_node][other_node]['weight'].remove(msps)
            except KeyError:
                pass

    def avg_deg(self):
        return self.number_of_edges() * 2 / self.num_nodes

    def deg_diff(self):
        return self.avg_deg() - self.target_deg

    def avg_wght(self):
        try:
            avg = mean([len(d['weight']) for u, v, d in self.edges(data=True)])
        except StatisticsError:
            avg = 0
        return avg

    def wght_diff(self):
        return self.avg_wght() - self.target_wght


if __name__ == '__main__':
    for i in range(1, 24):
        for j in range(1, 6):
            start = timer()
            g = MorphGraph(24, i, j)
            end = timer()
            print('=' * 79)
            print('=' * 79)
            print(end - start, 'seconds')
            print('deg:', i, g.avg_deg(), 'wght:', j, g.avg_wght(), sep='\t')
            print(g)
