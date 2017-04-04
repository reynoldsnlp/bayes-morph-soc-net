"""Implement agent-based social network model of Bayesian learning of morph."""

import random
import sys

from mesa import Agent
from mesa.datacollection import DataCollector as DC
from mesa import Model
from mesa.time import RandomActivation
import networkx as nx

SE = sys.stderr


def connect(gen_size, gen_count, nx_generator=nx.fast_gnp_random_graph,
            **kwargs):
    """Generate connections between agents."""
    er_networks = [[]] * gen_size
    for i in range(gen_count-1):
        # if i == 0:
        #     er_networks.extend([[]]*gen_size)
        #     continue
        er = nx_generator(**kwargs)
        # try:
        #     assert len(er) == gen_size * 2
        # except AssertionError:
        #     raise AssertionError('Network size does not match model'
        #                          'parameters\n'
        #                          '\texpected size:\t{}\n'
        #                          '\tactual size:\t{}\n'
        #                          '{}\n'
        #                          '{}'.format(gen_size * 2, len(er),
        #                                      er.adjacency_list(),
        #                                      len(er.adjacency_list())))
        relabel_dict = {j: j + (i * gen_size) for j in er}
        er = nx.relabel_nodes(er, relabel_dict)
        er_networks.extend(er.adjacency_list()[:gen_size])
    # TODO(RJR) ensure that all nodes have at least one connection?
    return er_networks


def homogen(model):
    """Return a model's homogeneity."""
    agent_morphs = [a.morphology for a in model.schedule.agents
                    if a.RIP is False]
    a_count = len([m for m in agent_morphs if m == 'a'])
    b_count = len([m for m in agent_morphs if m == 'b'])
    try:
        return a_count/b_count
    except ZeroDivisionError:
        return 0


def get_morph(agent):
    """Return agent's morphology."""
    return agent.morphology


def gen_morphs(N, proportionA):
    """Generate a random artificial morphology based on input parameters."""
    N = N*0.5
    a_count = round(N*proportionA)
    b_count = round(N) - a_count
    out_list = ['a']*a_count + ['b']*b_count
    random.shuffle(out_list)
    return out_list


class MorphAgent(Agent):
    """An agent to teach/learn a morphological grammar."""

    def __init__(self, unique_id, model, gen_id):
        """Initialize MorphAgent object."""
        print(',', unique_id, end="", file=SE)
        super().__init__(unique_id, model)
        self.gen_id = gen_id
        self.is_adult = False  # False=child, True=adult
        self.RIP = False  # agent has moved on to the great model in the sky
        self.morphology = ''
        self.connections = set(model.network[self.unique_id])
        self.input = []   # Input obtained from adult agents

    def __hash__(self):
        """Define MorphAgent's __hash__."""
        return hash((self.RIP, self.is_adult, self.unique_id))

    def step(self):
        """Take this action when called upon by the model's schedule."""
        print('Agent {:>5} is stepping...(gen_id:{:>2})'.format(self.unique_id,
                                                                self.gen_id),
              end='', file=SE)
        if self.is_adult and not self.RIP:
            print('  retiring...', file=SE)
            self.RIP = True
        elif not self.is_adult and self.model.schedule.steps == self.gen_id-1:
            print(' retrieving input...', end='', file=SE)
            # NB(RJR) Takes input from 'older' children of the same generation
            # Add conditional to match gen_id to remove intra-gen input
            print(' connections: {}'.format(self.connections), end='',
                  file=SE)
            self.input = [a.morphology for a in self.model.schedule.agents[:]
                          if a.is_adult and a.unique_id in self.connections]
            # print('self.input: {}'.format(self.input), file=SE)
            # inputs = []
            # for a in self.model.schedule.agents[:]:
            #     if a.is_adult:
            #         if a.unique_id in self.connections:
            #             inputs.append(a.morphology)
            print(' processing input...({})'.format(self.input), end='',
                  file=SE)
            self.process_input()  # Process input and generate output
            self.is_adult = True
        elif not self.is_adult and self.model.schedule.steps != self.gen_id-1:
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
        # adopt most frequent 'morphology' from input
        self.morphology = max(set(self.input), key=self.input.count)
        print('  ...done!', file=SE)


class MorphLearnModel(Model):
    """A multi-generation model with some number of agents."""

    def __init__(self, *, gen_size=25, gen_count=10, proportionA=0.5,
                 nw_adjacency_list=None,
                 nw_func=nx.fast_gnp_random_graph, nw_kwargs={'n': 50,
                                                              'p': 0.5}):
        """Initialize object."""
        print('Initializing model...', file=SE)
        nw_kwargs['n'] = gen_size*2  # override input value for network size
        self.num_agents = gen_size * gen_count
        print('  gen_size:', gen_size, file=SE)
        self.gen_size = gen_size
        print('  gen_count:', gen_count, file=SE)
        self.gen_count = gen_count
        self.proportionA = proportionA
        try:
            assert not nw_adjacency_list and nw_func
        except AssertionError:
            raise AssertionError('Model must initialize with either a network '
                                 'adjacency list or a generator function, not '
                                 'both.')
        if nw_adjacency_list:
            self.network = nw_adjacency_list
        else:
            print('Building networks with {}...'.format(nw_func.__name__),
                  file=SE)
            print('  kwargs:', nw_kwargs, file=SE)
            self.network = connect(gen_size, gen_count, nw_func, **nw_kwargs)
        print('='*79 + '\nNetwork adjacency list:\n', file=SE)
        for i, j in enumerate(self.network):
            print('{:>4} => {}'.format(i, j), file=SE)
        assert self.num_agents == len(self.network)
        self.schedule = RandomActivation(self)

        # Create agents
        print('Generating agents...', end='', file=SE)
        gen_counter = 0
        for i in range(self.num_agents):
            if i % self.gen_size == 0 and i > 0:
                gen_counter += 1
            a = MorphAgent(i, self, gen_counter)
            if i < self.gen_size:  # The 1st generation are adults
                a.is_adult = True
                a.morphology = random.choice(['a', 'b'])
                print(a.morphology, end='', file=SE)
            self.schedule.add(a)
        print(file=SE)

        self.dc = DC(model_reporters={'Homogen': homogen},
                     agent_reporters={'Morph': get_morph})

    def step(self):
        """Advance the model by one step."""
        print('Model is stepping...', file=SE)
        self.dc.collect(self)  # collect data
        self.schedule.step()
