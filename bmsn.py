"""Implement agent-based social network model of Bayesian learning of morph."""

import itertools
import random
import sys

from mesa import Agent
from mesa.datacollection import DataCollector as DC
from mesa import Model
from mesa.time import RandomActivation


def homogen(model):
    """Return a model's homogeneity."""
    agent_morphs = [a.morphology for a in model.schedule.agents
                    if a.RIP is False]
    a_count = len([m for m in agent_morphs if m == 'a'])
    b_count = len([m for m in agent_morphs if m == 'b'])
    # N = model.num_agents
    try:
        return a_count/b_count
    except ZeroDivisionError:
        return 0


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

    def __init__(self, unique_id, model):
        """Initialize MorphAgent object."""
        print(',', unique_id, end="", file=sys.stderr)
        super().__init__(unique_id, model)
        self.is_adult = False  # False=child, True=adult
        self.RIP = False  # agent has moved on to the great model in the sky
        self.morphology = ''
        self.adult_connections = []  # Adult agents from whom input is obtained
        self.input = []   # Input obtained from adult agents
        self.output = []  # Output produced for child agents

    def __hash__(self):
        """Define MorphAgent's __hash__."""
        return hash((self.RIP, self.is_adult, self.unique_id))

    def step(self):
        """Take this action when called upon by the model's schedule."""
        print('Agent {:>5} is stepping...'.format(self.unique_id), end='',
              file=sys.stderr)
        if self.is_adult and not self.RIP:
            self.RIP = True
            print(' retired...', file=sys.stderr)
        elif not self.is_adult:
            print(' processing input...({})'.format(self.input),
                  file=sys.stderr)
            self.process_input()  # Process input and generate output
            self.is_adult = True
        elif self.RIP:
            # raise RuntimeError('Agent {:>5} is already '
            #                    'retired.'.format(self.unique_id))
            print( 'Agent {:>5} is already retired.'.format(self.unique_id))
        else:
            raise RuntimeError('Something strange with agent '
                               '{:>5}.'.format(self.unique_id))

    def process_input(self):
        """Do something interesting, but Bayesian."""
        # adopt most frequent 'morphology' from input
        self.morphology = max(set(self.input), key=self.input.count)
        self.output = self.morphology


class MorphLearnModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, proportionA=0.5, network='poisson', p=0.5):
        """Initialize object."""
        print('Initializing model...', file=sys.stderr)
        self.num_agents = N
        self.num_children = round(N/2)
        self.proportionA = proportionA
        self.network = network
        self.p = p
        self.seed_morphs = gen_morphs(N, proportionA)
        self.id_generator = itertools.count(0)
        self.schedule = RandomActivation(self)

        # Create agents
        print('Generating agents...', end='', file=sys.stderr)
        for i in range(self.num_agents):
            a = MorphAgent(self.id_generator.__next__(), self)
            if i < self.num_children:  # The 1st half are adults
                a.is_adult = True
                a.morphology = random.choice(['a', 'b'])
                print(a.morphology, end='', file=sys.stderr)
            self.schedule.add(a)
        print('agents type: {}'.format(type(self.schedule.agents)))
        print(file=sys.stderr)
        self.connect()

        self.dc = DC(model_reporters={'Homogen': homogen},
                     agent_reporters={'Morph': lambda a: a.morphology})

    def step(self):
        """Advance the model by one step."""
        print('Model is stepping...', file=sys.stderr)
        self.dc.collect(self)
        self.schedule.step()
        # Generate new child agents
        print('Generating next generation of agents...', file=sys.stderr)
        for i in range(self.num_children):
            a = MorphAgent(self.id_generator.__next__(), self)
            self.schedule.add(a)
        print(file=sys.stderr)
        print(type(self.schedule.agents), self.schedule.agents,
              dir(self.schedule.agents))
        assert all([not agent.is_adult
                    for agent in self.schedule.agents[-1*self.num_children:]])
        self.connect()

    def connect(self, network='E-R', p=0.1):
        """Generate connections between agents and transfer input to child."""
        if network == 'E-R':  # 'Erdos-Renyi' aka 'poisson'
            print('Building Erdos-Renyi network...', file=sys.stderr)
            # p = p/float(self.num_agents)
            # 'k' (kappa) is from options.avgNbrs
            for c in self.schedule.agents[(-1*self.num_children):]:
                # Make sure that every child has at least one connection
                parent = random.choice(self.schedule.agents[(-1*self.num_agents):(-1*self.num_children)])
                c.adult_connections.append(parent.unique_id)
                c.input.append(parent.morphology)
                for a in self.schedule.agents[(-1*self.num_agents):(-1*self.num_children)]:
                    if random.random() <= p:
                        c.adult_connections.append(a.unique_id)
                        c.input.append(a.morphology)
                        print('Making connection between agents {:>5}   and '
                              '{:>5}...'.format(c.unique_id, a.unique_id),
                              file=sys.stderr)
