"""Implement new scheduler with multiprocessing."""


class MultiScheduler:
    """Like BaseScheduler, but with multiprocessing.

    Activates agents one at a time, in the order they were added.
    Assumes that each agent added has a *step* method which takes no arguments.
    """

    def __init__(self, model, j=None):
        """Create a new, empty BaseScheduler."""
        from multiprocessing import Pool
        if j is None:
            try:
                import subprocess as sp

                p = sp.run('nproc', shell=True, check=True, stdout=sp.PIPE)
                self.j = int(p.stdout)
                print(f'Number of cores set to {self.j}')
            except:
                print('Problem determining number of cores; default=1')
                self.j = 1
        else:
            self.j = j  # number of processes
        self.pool = Pool(j)
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []

    def __getstate__(self):
        """Remove `self.pool` for pickling."""
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        """Restore `self.__dict__` after unpickling."""
        self.__dict__.update(state)

    def add(self, agent):
        """Add an Agent object to the schedule.

        Args:
            agent: An Agent to be added to the schedule. NOTE: The agent must
            have a step() method.
        """
        self.agents.append(agent)

    def remove(self, agent):
        """Remove all instances of a given agent from the schedule.

        Args:
            agent: An agent object.
        """
        while agent in self.agents:
            self.agents.remove(agent)

    def step(self):
        """Execute the step of all the agents, one at a time."""
        self.pool.map(self.pool_step, self.agents[:])
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """Return the current number of agents in the queue."""
        return len(self.agents)

    def pool_step(self, agent):
        """Execute step method of `agent`."""
        agent.step()
