
from lexicon import Lexicon
from bisect import bisect_left
import language
import random, numpy, codecs
random.seed()

class Agent:
	def __init__(self, attributes = {}):
		self.attributes = attributes or {}
		self.lexicon = Lexicon()
		self.grammar = {}

	def read(self, fileName):	# updates lexicon with input files
		try: self.lexicon.read(fileName)
		except IOError: print >> sys.stderr, "Couldn't open file %s" %fileName

	def listen(self, lex):
		self.lexicon.update(lex)

	def learn(self):
		"""
This function builds grammar from input lexicon
The 'dist' data structure starts as raw counts from the lexicon
	specifically, it is initialized as a [function]x[concept] tensor of zeroes
	it is then filled from the lexicon data structure
Then, it is factored into lemma frequency and function probabilities
	e.g. [30 20 20 10 10 10] = [60]*[.3 .2 .2 .1 .1 .1] 
Estimation of the MAP hypothesis -- learning -- consists of postmultiplication by the sims matrix
	the sims matrix mixes the raw data with a "prior"
	the prior is a similarity-weighted averaged over a random sample of existing forms
	it is column-normalized to sum to 1, which keeps dist a pdf
	in the case of uniform weighting, the effect is to pull low-frequency lemmas toward the mean
	if the weighting scheme is more complex, correspondingly more complex behavior can be obtained
		in particular, forms can be preferentially attracted to one "gang" over another
		for example, on the basis of morphophonological similarity
Finally, the resulting data structures are stored as attributes of the agent
		"""
	
		## get list of concepts, functions; sort for efficient indexing
		conceptIndex, concepts = self.lexicon.fields.index(u'CONCEPT'), sorted(self.lexicon.project(u'CONCEPT'))
		funcIndex, functions = self.lexicon.fields.index(u'FUNCTION'), sorted(self.lexicon.project(u'FUNCTION'))
		self.attributes['conceptList'], nConcepts = concepts, len(concepts)
		self.attributes['functionList'], nFuncs = functions, len(functions)

		## set learning parameters
		kappa = min(nConcepts,self.attributes['kappa'])		# sample size for computing prior
		beta = float(self.attributes['beta'])/kappa		# set learning parameter

		## the phattie data structures
		dist = numpy.zeros((nFuncs, nConcepts),float)

		## iterate through entries in lexicon and dump to counts
		for entry in self.lexicon:
			iFunction = functions.index(entry[funcIndex])
			jConcept = bisect_left(concepts, entry[conceptIndex])
			dist[iFunction, jConcept] = float(self.lexicon[entry])

		## factor counts into frequency and function-distribution
		freq = numpy.add.reduce(dist)					## for each concept, sum across functions to get lemma frequency
		numpy.divide(dist, freq, dist)					## divide raw counts by lemma frequency to get sample probability
		self.attributes['conceptFreqs'] = numpy.add.accumulate(freq)	## store as cumulative pdf, but keep as pdf here

		## calculating summation matrix
		sims = numpy.zeros((nConcepts,nConcepts),float)
		for jTarg in xrange(nConcepts):
			## prior is sample from other forms (from raw others), weighted by beta
			sampleIndices = random.sample(xrange(nConcepts), kappa)
			for iSource in sampleIndices:
				sims[iSource,jTarg] = beta*self.attributes['simFunction'](concepts[jTarg],concepts[iSource])
			## mix prior with observed data (from raw self), weight = own frequency
			sims[jTarg,jTarg] = freq[jTarg]

		## normalize summation matrix (automatic linear interpolation!)
		numpy.divide(sims, numpy.add.reduce(sims), sims)

		## calculating posterior by postmultiplication with summation matrix, store to self.attributes
		self.attributes['grammar'] = numpy.matrix(dist) * sims

	def talk(self, nUtts):
		if not self.attributes.has_key('grammar'):	# if no grammar, produce by random sampling from input lexicon
			return(self.lexicon.sample(nUtts))

		sample = Lexicon(fields = self.lexicon.fields)	# data structure to be returned: a Lexicon object
		grammar = self.attributes['grammar']		# grammar object created by learn()
		concepts = self.attributes['conceptList']	# list of concepts (string objects)
		cumFreq = self.attributes['conceptFreqs']	# cumulative frequency distribution of lemma frequencies
		freqMass = cumFreq[-1]				# total frequency mass (final entry of cumuluative freq dist)
		nConcepts = len(concepts)			# number of lemma types
		functions = self.attributes['functionList']	# list of functions

		for iUtt in xrange(int(nUtts)):
			jConcept = bisect_left(cumFreq, freqMass*random.random())
			iFunc = bisect_left(numpy.add.accumulate(grammar[:,jConcept]), random.random())
			key = (concepts[jConcept], functions[iFunc], concepts[jConcept]+language.infl[functions[iFunc]])
			if sample.has_key(key): sample[key] += 1
			else: sample[key] = 1

		return(sample)

	def store(self, fileName = "grammar.txt", encoding='utf-8'):
		# saves agent's grammar
		if not self.attributes.has_key('grammar'): return
		functions = self.attributes['functionList']
		concepts = self.attributes['conceptList']
		freq = self.attributes['conceptFreqs']
		grammar = self.attributes['grammar']
		fGram = codecs.open(fileName, 'wt', encoding)

		for function in functions: fGram.write('\t%s' %function)
		fGram.write('\n')

		for jConcept in xrange(len(concepts)):
			fGram.write('%s\t%d' %(concepts[jConcept],freq[jConcept]))
			for iFunc in xrange(len(functions)): fGram.write('\t%s' %grammar[iFunc, jConcept])
			fGram.write('\n')
		fGram.close()

