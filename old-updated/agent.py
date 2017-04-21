from bisect import bisect_left
import numpy
import random
import sys
random.seed()

import language
from lexicon import Lexicon


class Agent:
    """Agent in the model."""

    def __init__(self, attributes={}):
        """Initialize Agent object."""
        self.attributes = attributes or {}
        self.lexicon = Lexicon()
        self.grammar = {}

    def read(self, fileName):
        """Update lexicon with input files."""
        try:
            self.lexicon.read(fileName)
        except IOError:
            print("Couldn't open file {}".format(fileName), file=sys.stderr)

    def listen(self, lex):
        """Update lexicon based on input lexicon."""
        self.lexicon.update(lex)

    def learn(self):
        """Build grammar from input lexicon.

        The 'dist' data structure starts as raw counts from the
        lexicon specifically. It is initialized as a
        [function]x[concept] tensor of zeroes. It is then filled from
        the lexicon data structure. Then, it is factored into lemma
        frequency and function probabilities, e.g.
        [30 20 20 10 10 10] = [60]*[.3 .2 .2 .1 .1 .1]

        Estimation of the MAP hypothesis -- learning -- consists of
        postmultiplication by the sims matrix. The sims matrix mixes
        the raw data with a "prior". The prior is a similarity-weighted
        averaged over a random sample of existing forms. It is
        column-normalized to sum to 1, which keeps dist a pdf in the
        case of uniform weighting. The effect is to pull low-frequency
        lemmas toward the mean if the weighting scheme is more complex.
        Correspondingly more complex behavior can be obtained. In
        particular, forms can be preferentially attracted to one "gang"
        over another. For example, on the basis of morphophonological
        similarity. Finally, the resulting data structures are stored
        as attributes of the agent.
        """
        # get list of concepts, functions; sort for efficient indexing
        conceptIndex = self.lexicon.fields.index('CONCEPT')
        concepts = sorted(self.lexicon.project('CONCEPT'))
        funcIndex = self.lexicon.fields.index('FUNCTION')
        functions = sorted(self.lexicon.project('FUNCTION'))
        self.attributes['conceptList'] = concepts
        nConcepts = len(concepts)
        self.attributes['functionList'] = functions
        nFuncs = len(functions)

        # set learning parameters
        # kappa = sample size for computing prior
        kappa = min(nConcepts, self.attributes['kappa'])
        # beta = learning parameter
        beta = float(self.attributes['beta'])/kappa

        # the phattie data structures
        dist = numpy.zeros((nFuncs, nConcepts), float)

        # iterate through entries in lexicon and dump to counts
        for entry in self.lexicon:
            iFunction = functions.index(entry[funcIndex])
            jConcept = bisect_left(concepts, entry[conceptIndex])
            dist[iFunction, jConcept] = float(self.lexicon[entry])

        # factor counts into frequency and function-distribution
        # for each concept, sum across functions to get lemma frequency
        freq = numpy.add.reduce(dist)
        # divide raw counts by lemma frequency to get sample probability
        numpy.divide(dist, freq, dist)
        # store as cumulative pdf, but keep as pdf here
        self.attributes['conceptFreqs'] = numpy.add.accumulate(freq)

        # calculating summation matrix
        sims = numpy.zeros((nConcepts, nConcepts), float)
        for jTarg in range(nConcepts):
            # prior is sample from other raw forms, weighted by beta
            sampleIndices = random.sample(range(nConcepts), kappa)
            for iSource in sampleIndices:
                sims[iSource, jTarg] = (beta
                                        * self.attributes['simFunction'](concepts[jTarg],
                                                                         concepts[iSource]))
            # mix prior with observed data (from raw self), weight = own freq
            sims[jTarg, jTarg] = freq[jTarg]

        # normalize summation matrix (automatic linear interpolation!)
        numpy.divide(sims, numpy.add.reduce(sims), sims)

        # calculate posterior by postmultiplication with summation matrix
        # store to self.attributes
        self.attributes['grammar'] = numpy.matrix(dist) * sims

    def talk(self, nUtts):
        """Generate output strings."""
        # if no grammar, produce by random sampling from input lexicon
        if 'grammar' not in self.attributes:
            return self.lexicon.sample(nUtts)

        sample = Lexicon(fields=self.lexicon.fields)
        grammar = self.attributes['grammar']       # object created by learn()
        concepts = self.attributes['conceptList']  # list of concepts strings
        cumFreq = self.attributes['conceptFreqs']  # cumulative lemma freq dist
        freqMass = cumFreq[-1]                     # total frequency mass
        nConcepts = len(concepts)                  # number of lemma types
        functions = self.attributes['functionList']  # list of functions

        for iUtt in range(int(nUtts)):
            jConcept = bisect_left(cumFreq, freqMass*random.random())
            iFunc = bisect_left(numpy.add.accumulate(grammar[:, jConcept]),
                                random.random())
            key = (concepts[jConcept], functions[iFunc],
                   concepts[jConcept]+language.infl[functions[iFunc]])
            if key in sample:
                sample[key] += 1
            else:
                sample[key] = 1

        return sample

    def store(self, fileName='grammar.txt', encoding='utf-8'):
        """Save agent's grammar."""
        if 'grammar' not in self.attributes:
            return None
        functions = self.attributes['functionList']
        concepts = self.attributes['conceptList']
        freq = self.attributes['conceptFreqs']
        grammar = self.attributes['grammar']
        fGram = open(fileName, 'wt', encoding=encoding)

        for function in functions:
            fGram.write('\t{}'.format(function))
        fGram.write('\n')

        for jConcept in range(len(concepts)):
            fGram.write('{}\t{}'.format(concepts[jConcept], freq[jConcept]))
            for iFunc in range(len(functions)):
                fGram.write('\t{}'.format(grammar[iFunc, jConcept]))
            fGram.write('\n')
        fGram.close()
