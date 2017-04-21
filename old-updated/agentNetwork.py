"""Provide bare minimum functionality of multi-gen social network."""

import random


class AgentNetwork:
    """Define AgentNetwork object.

    Each agent is equipped with an index and a rank.
    'rank' corresponds to an age grade, e.g. 0 = children; 1 = adults.
    'index' is an arbitrary value. It indicates the agent's canonical
    position when the list of agents at a given rank is requested.

    adjList is a list-of-lists. If adjList[i] = [j1, j2], it means that
    there are outgoing links from agents j1 and j2 to agent i.

    In the present implementation, the code relies on implicit
    rank-coding. Agents j1 and j2 belong to the adult (rank=1)
    generation, whereas agent i belongs to the child (rank=0) gen'n.
    """

    def __init__(self):
        """Initialize agent network object."""
        self.adjList = {}   # adjList[agentID] = [IDs, of, incoming, links]
        self.agentDirectory = {}

    def birth(self, n, agentType, agentParams):
        """TODO(RJR) I don't know what this does yet."""
        nextIndex = 0
        for rank, iAgent in self.agentDirectory:
            if rank == 0:
                nextIndex = max(iAgent+1, nextIndex)
        for iAgent in range(nextIndex, nextIndex+n):
            self.agentDirectory[(0, iAgent)] = agentType(attributes=agentParams)

    def killRank(self, killrank=1):
        """Kill agents of a particular rank (default=1)."""
        for agentKey in self.agentDirectory:
            rank = agentKey[0]
            if rank == killrank:
                del self.agentDirectory[agentKey]

    def getRank(self, targrank=0):
        """TODO(RJR) I don't know what this does yet."""
        outorder = []
        for agentKey in self.agentDirectory:
            rank, iAgent = agentKey[0], agentKey[1]
            if rank == targrank:
                outorder.append([iAgent, self.agentDirectory[agentKey]])
        outorder.sort()
        return [index_agent[1] for index_agent in outorder]

    def promote(self):
        """Move agents up 1 rank."""
        maxRank = max([agentKey[0] for agentKey in self.agentDirectory])
        while maxRank >= 0:
            agentList = self.getRank(maxRank)
            for iAgent, agent in enumerate(agentList):
                del self.agentDirectory[(maxRank, iAgent)]
                self.agentDirectory[(maxRank+1, iAgent)] = agent
            maxRank -= 1


def connect(n, method='poisson', type=list, connectParams=None):
    """Return some representation of a social network/graph.

    Major variations involve method and type

    TYPE
        list -- retval[i] = list of indices that connect to i
        dict -- all tuples (i,j) for which j connects to i

    METHOD
        poisson -- j connected to i with probability p = k/n
            where k is average number of neighbors
            and n is number of agents in network
            REQUIRES PARAMETER 'k'
    """
    if method == 'poisson':
        p = connectParams['k']/float(n)  # 'k' (kappa) is from options.avgNbrs
        if type == list:
            retval = [[] for i in range(n)]
        elif type == dict:
            retval = {}
        for i in range(n):
            for j in range(n):
                if random.random() < p:
                    if type == list:
                        retval[i].append(j)
                    elif type == dict:
                        retval[(i, j)] = 1
        return retval
    return None
