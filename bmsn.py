"""Implement agent-based social network model of Bayesian learning of morph."""

import random
from statistics import mean
import sys

from mesa import Agent
from mesa.datacollection import DataCollector as DC
from mesa import Model
from mesa.time import RandomActivation

SE = sys.stderr


def connect(gen_size, gen_count, nx_gen, **kwargs):
    """Generate connections between agents."""
    if nx_gen is None:
        import networkx as nx
        nx_gen = nx.fast_gnp_random_graph
    if kwargs == {}:
        kwargs = {'p': 0.5}
    kwargs['n'] = gen_size * 2  # override input value for network size
    print('Building networks with {}...'.format(nx_gen.__name__), file=SE)
    print('  kwargs:', kwargs, file=SE)
    er_networks = [[]] * gen_size  # 1st gen does not have connections
    for i in range(gen_count - 1):
        er = nx_gen(**kwargs)
        relabel_dict = {j: j + (i * gen_size) for j in er}
        er = nx.relabel_nodes(er, relabel_dict)
        er_networks.extend(er.adjacency_list()[:gen_size])
    # TODO(RJR) ensure that all nodes have at least one connection?
    return er_networks


def N2p(in_dict):
    """Transform raw count dict to proportional dict."""
    N = sum(in_dict.values())
    return {k: v / N for k, v in in_dict.items()}


def dicts2dict(list_of_dicts):
    """Enforce coherence of lexeme by averaging probs across conditions."""
    print('dicts2dict...{}'.format(list_of_dicts), file=SE)
    keys = list_of_dicts[0].keys()
    assert all([keys == i.keys() for i in list_of_dicts])
    return {k: mean([i[k] for i in list_of_dicts]) for k in keys}


def discrete_output(in_dict):
    """Return the key with the highest value."""
    return max(in_dict.items(), key=lambda x: x[1])[0]


def prob_output(in_dict):
    """Return a random key, choice weighted by value.

    Given {0: 0.1, 1: 0.9} 1 has a 90 percent chance of being selected.
    """
    keys = list(in_dict.keys())
    return random.choices(keys, weights=[in_dict[k] for k in keys])


def homogen(model):
    """Return a model's homogeneity."""
    agent_morphs = [a.morphology for a in model.schedule.agents
                    if a.RIP is False]
    a_count = len([m for m in agent_morphs if m == 'a'])
    b_count = len([m for m in agent_morphs if m == 'b'])
    try:
        return a_count / b_count
    except ZeroDivisionError:
        return 0


def get_morph(agent):
    """Return agent's morphology."""
    return agent.morphology


def gen_morphs(N, proportionA):
    """Generate a random artificial morphology based on input parameters."""
    N = N * 0.5
    a_count = round(N * proportionA)
    b_count = round(N) - a_count
    out_list = (['a'] * a_count) + (['b'] * b_count)
    random.shuffle(out_list)
    return out_list


class MorphAgent(Agent):
    """An agent to teach/learn a morphological grammar."""

    def __init__(self, unique_id, model, gen_id):
        """Initialize MorphAgent object."""
        print('{}, '.format(unique_id), end="", file=SE)
        super().__init__(unique_id, model)
        self.gen_id = gen_id
        self.is_adult = False  # False=child, True=adult
        self.RIP = False  # agent has moved on to the great model in the sky
        self.morphology = ''
        self.connections = set(model.network[self.unique_id])
        self.input = []   # Input obtained from adult agents
        self.ddist = {}

    def __hash__(self):
        """Define MorphAgent's __hash__ for sorting."""
        return hash((self.RIP, self.is_adult, self.unique_id))

    def step(self):
        """Take this action when called upon by the model's schedule."""
        print('Agent {:>5} is stepping...(gen_id:{:>2})'.format(self.unique_id,
                                                                self.gen_id),
              end='', file=SE)
        if self.is_adult and not self.RIP:  # 'active' adults
            print('  retiring...', file=SE)
            self.RIP = True
        elif (not self.is_adult and
              self.model.schedule.steps == self.gen_id - 1):  # active children
            print('\n\t\t\tretrieving input...', end='', file=SE)
            print('\n\t\t\tconnections: {}'.format(self.connections), end='',
                  file=SE)
            new_input = [a.speak() for a in self.model.schedule.agents[:]
                         if a.is_adult and
                         a.unique_id in self.connections and
                         a.gen_id != self.gen_id]
            for i in new_input:
                self.input.extend(i)
            self.input = sorted(self.input)
            # print('self.input: {}'.format(self.input), file=SE)
            # inputs = []
            # for a in self.model.schedule.agents[:]:
            #     if a.is_adult:
            #         if a.unique_id in self.connections:
            #             inputs.append(a.morphology)
            print('\n\t\t\tprocessing input...{}...'.format(self.input[:2]),
                  end='', file=SE)
            self.process_input()  # Process input and generate output
            self.is_adult = True
        elif (not self.is_adult and
              self.model.schedule.steps != self.gen_id - 1):
            print('  still unborn.', file=SE)
        elif self.RIP:
            # raise RuntimeError('Agent {:>5} is already '
            #                    'retired.'.format(self.unique_id))
            print('  already retired.'.format(self.unique_id), file=SE)
        else:
            raise RuntimeError('Something strange with agent '
                               '{:>5}.'.format(self.unique_id))

    def process_input(self):
        """Do something interesting, but Bayesian."""
        # data distribution
        print('\n\t\t\tcompiling data distribution...', file=SE)
        ddist = {}  # ddist[(l, t_ms, g_ms, g_e)] = {e1: x, e2: y, ...}
        for l, t_ms, t_e, g_ms, g_e in self.input:
            try:
                ddist[(l, t_ms, g_ms, g_e)][t_e] += 1
            except KeyError:
                try:
                    ddist[(l, t_ms, g_ms, g_e)][t_e] = 1
                except KeyError:
                    ddist[(l, t_ms, g_ms, g_e)] = {t_e: 1}
        # transform counts to probabilities
        ddist = {k: N2p(v) for k, v in ddist.items()}
        # lexical dictionary
        # lex_dict[(l, t_ms)] = {e1: p1, e2: p2, ...}
        print('\t\t\tcompiling lex_dict...', file=SE)
        # l_list = sorted(list(set([(l, t_ms) for l, t_ms, g_ms, g_e in ddist])))  # noqa
        l_list = sorted(list(set([k[0] for k in ddist])))
        lex_dict = {}
        for l in l_list:
            lex_dict[l] = {}
            for t_ms in self.model.seed_MSPSs:
                lex_dict[l][t_ms] = dicts2dict([v for k, v in ddist.items()
                                                if k[:2] == (l, t_ms)])
        self.lex_dict = lex_dict
        print('\t\t\t...done!', file=SE)

    def speak(self):
        """Generate output.

        Each datum is a tuple in the following format:
        (lexeme, target_msps, target_ending, given_msps, given_ending)

        Return a list of tuples.
        """
        if self.input == []:
            return [i for i in self.model.seed_data() if random.random() > 0.5]
            # TODO(RJR) parametrize 0.5 in previous line
        else:
            out_list = []
            for l, d in self.lex_dict.items():
                for t_ms in d:
                    for g_ms in d:
                        if t_ms != g_ms and random.random() > 0.5:
                            # TODO(RJR) parametrize 0.5 in previous line
                            out_list.append((l,
                                             t_ms,
                                             self.model.out_func(d[t_ms]),
                                             g_ms,
                                             self.model.out_func(d[g_ms])))
            return out_list


class MorphLearnModel(Model):
    """A multi-generation model with some number of agents."""

    def __init__(self, *, gen_size=25, gen_count=10, morph_filename=None,
                 nw_func=None, nw_kwargs={}, discrete=True, whole_lex=True):
        """Initialize model object.

        Arguments:
        gen_size  -- number of agents per generation
        gen_count -- number of generations to simulate
        morph_filename  -- filename of tab-sep table: msps x classes
        nw_func   -- One of the following types...
            list     -- e.g. list[0] is list of connected nodes
            function -- function to generate adjacency list
            A function must be accompanied by nw_kwargs!
        nw_kwargs -- dict of arguments for nw_func function
        discrete -- boolean: discretize probability before productions
        whole_lex -- boolean: use whole lexicon to calculate priors
                     ...or else a random sampling of neighbors.
        """
        print('Initializing model...', file=SE)
        self.num_agents = gen_size * gen_count
        print('  gen_size:', gen_size, file=SE)
        self.gen_size = gen_size
        print('  gen_count:', gen_count, file=SE)
        self.gen_count = gen_count
        self.parse_seed_morph(morph_filename)
        # try:
        self.network = connect(gen_size, gen_count, nw_func, **nw_kwargs)
        # except TypeError:
        #     try:
        #         assert isinstance(nw_func, list)
        #     except AssertionError:
        #         raise AssertionError('nw_func must be either a function or '
        #                              'an adjacency list.')
        #     self.network = nw_func
        print('=' * 79, file=SE)
        print('Network adjacency list:\n', file=SE)
        for i, j in enumerate(self.network):
            print('{:>4} => {}'.format(i, j), file=SE)
        assert self.num_agents == len(self.network)
        self.discrete = discrete
        if self.discrete:
            self.out_func = discrete_output
        else:
            self.out_func = prob_output
        self.whole_lex = whole_lex
        self.schedule = RandomActivation(self)  # TODO(RJR) rand unnecessary?

        # Create agents
        print('Generating agents...', file=SE)
        gen_counter = 0
        for i in range(self.num_agents):
            if i % self.gen_size == 0 and i > 0:
                gen_counter += 1
            a = MorphAgent(i, self, gen_counter)
            if i < self.gen_size:  # The 1st generation are adults
                a.is_adult = True
            self.schedule.add(a)
        print(file=SE)

        # Data collectors
        self.dc = DC(model_reporters={'Homogen': homogen},
                     agent_reporters={'Morph': get_morph})

    def parse_seed_morph(self, input_filename):
        """Build seed morphology from a file.

        The first row contains headers, 'typeFreq' followed by MSPSs:
        typeFreq    MSPS1   MSPS2   MSPS3   etc.    ...

        Each following row represents an inflection class:
        457         a       a       i       etc.    ...
        12          a       i       i       etc.    ...
        ...
        ...
        """
        # TODO(RJR) Make all of this part of a Seed object?
        self.seed_filename = input_filename
        with open(input_filename) as in_file:
            self.seed_cols = [c for c
                              in in_file.readline().rstrip().split('\t')]
            self.seed_MSPSs = self.seed_cols[1:]
            self.seed_infl_classes = []  # list of dicts
            for line in in_file:
                infl_class = {}
                for i, value in enumerate(line.rstrip().split('\t')):
                    if self.seed_cols[i] == 'typeFreq':
                        infl_class[self.seed_cols[i]] = int(value)
                    else:
                        try:
                            infl_class[self.seed_cols[i]] = float(value)
                        except ValueError:
                            infl_class[self.seed_cols[i]] = value
                self.seed_infl_classes.append(infl_class)
        self.seed_msps_dict = {}
        for msps in self.seed_MSPSs:
            self.seed_msps_dict[msps] = {}
            for i_class in self.seed_infl_classes:
                try:
                    self.seed_msps_dict[msps][i_class[msps]] += i_class['typeFreq']  # noqa
                except KeyError:
                    self.seed_msps_dict[msps][i_class[msps]] = i_class['typeFreq']  # noqa
        self.seed_mnb_dict = {}
        for target_msps in self.seed_MSPSs:
            for given_msps in self.seed_MSPSs:
                if target_msps != given_msps:
                    for given_ending in self.seed_msps_dict[given_msps]:
                        tgg = (target_msps, given_msps, given_ending)
                        self.seed_mnb_dict[tgg] = self.seed_MNBs(*tgg)

    def seed_MNBs(self, target_msps, given_msps, given_ending):
        """Calculate type-freq-weighted prevalence of endings.

        Given an MSPS and its ending, return the probability of each
        ending for a particular MSPS, e.g...

        NomSg | (AccSg = -u) -> {ø: 0.0, a: 1.0, o: 0.0}

        Return dictionary of ending:probability pairs.
        """
        out_dict = {}
        for i in self.seed_infl_classes:
            if i[given_msps] == given_ending:
                try:
                    out_dict[i[target_msps]] += i['typeFreq']
                except KeyError:
                    out_dict[i[target_msps]] = i['typeFreq']
        return N2p(out_dict)

    def MNBs(self, lex_dict, target_msps, given_msps, given_ending):
        """Calculate type-freq-weighted prevalence of endings.

        Given an MSPS and its ending, return the probability of each
        ending for a particular MSPS, e.g...

        NomSg | (AccSg = -u) -> {ø: 0.0, a: 1.0, o: 0.0}

        Return dictionary of ending:probability pairs.
        """
        out_dict = {}
        for l, d in lex_dict.items():
            if self.out_func(d[given_msps]) == given_ending:
                try:
                    out_dict[self.out_func(d[target_msps])] += 1
                except KeyError:
                    out_dict[self.out_func(d[target_msps])] = 1
        return N2p(out_dict)

    def seed_data(self):
        """Deterministic generator for 1st-gen 'speech' production.

        Each datum is a tuple in the following format:
        (lexeme, target_msps, target_ending, given_msps, given_ending)
        """
        for ci, c in enumerate(self.seed_infl_classes):
            for i in range(c['typeFreq']):  # each lexeme is named ci-i
                for m in self.seed_MSPSs:
                    for n in self.seed_MSPSs:
                        if m != n:
                            yield ('{}-{}'.format(ci, i), m, c[m], n, c[n])

    def step(self):
        """Advance the model by one step."""
        print('Model is stepping...', file=SE)
        self.dc.collect(self)  # collect data
        self.schedule.step()
