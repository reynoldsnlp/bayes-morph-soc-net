import codecs
from random import random
from bisect import bisect
import sys

class Lexicon(dict):								
	def __init__(self, fileName = None, fields = None, name = 'lexicon', data = None):
		self.name = name			# give lexicon a name
		self.fields = fields or []
		if fileName: self.read(fileName)	# read in from a file
		self.nFields = len(self.fields)		# number of fields
		if data: self.update(data) 		# merge new data
		
	def select(self, subset = {}):
		## returns the keys to the subset of the lexicon that matches the
		##	structural description of the query (dictionary) object, e.g.
		## lex.query({'AGENT': 'Ivan'})
		##	returns all the keys of lex whose AGENT field has the value Ivan
		subsetKeys = self.keys()
		for field in subset:
			fieldIndex = self.fields.index(field)
			keepVals = subset[field]
			keepKeys = []
			for key in subsetKeys:
				if key[fieldIndex] in keepVals: keepKeys.append(key)
			subsetKeys = keepKeys[:]
		return(subsetKeys)

	def fromKeys(self, subset):
		## returns the sublexicon whose keys are selected by subset
		## probably this means shallow copy for mutables data structures
		subsetKeys = self.select(subset)
		destLex = Lexicon(fields = self.fields)
		for key in subsetKeys:
			destLex[key] = self[key]
		return(destLex)

	def aggregate(self, aggrFields = [], subset = {}):
		# aggregates subset over aggrFields, e.g. aggregate(["CONCEPTS", "MORPHO"]) collects
		#   population-level stats on concept-morphosyntax pairs
		# can be combined with subsetting to strip indices, e.g. aggregate(["CONCEPTS",
		#   "MORPHO", "FORM"], {"AGENT": ["Ivan"]}) yields Ivan's lexicon, stripped
		#   of the AGENT label 

		aggrLex = Lexicon(fields = aggrFields)
		keepIndices = [self.fields.index(field) for field in self.fields if field in aggrFields]
		sumIndices = [self.fields.index(field) for field in self.fields if field not in aggrFields]
		subsetKeys = self.select(subset)
		for key in subsetKeys:
			if len(keepIndices) > 1: aggrKey = tuple([key[index] for index in keepIndices])
			else: aggrKey = key[keepIndices[0]]
			if aggrLex.has_key(aggrKey): aggrLex[aggrKey] += self[key]
			else: aggrLex[aggrKey] = self[key]
		return(aggrLex)

	def sample(self, nSamps = 1, subset = {}):
		sample = Lexicon(fields = self.fields)
		keys = self.select(subset)
		if len(keys) == 0: return(sample)

		## set up cumulative distribution function
		cumPdf = [0]
		for key in keys:
			cumPdf.append(cumPdf[-1]+self[key])

		## map random numbers to sample
		for iSamp in xrange(nSamps):
			key = keys[bisect(cumPdf,cumPdf[-1]*random())-1]
			if sample.has_key(key): sample[key] += 1
			else: sample[key] = 1

		return(sample)

	def bind(self, sourceLex, bindings, merge = True):
		for field in self.fields:
			if field not in sourceLex.fields and not bindings.has_key(field):
				raise TypeError('Lexicon missing key field %s\n' % field)
		for sourceKey in sourceLex:
			lKey = []
			for field in self.fields:
				if bindings.has_key(field): lKey.append(bindings[field])
				else: lKey.append(sourceKey[sourceLex.fields.index(field)])
			key = tuple(lKey)
			if merge or not self.has_key(key): self[key] = sourceLex[sourceKey]
			else:
				self[key] = self[key] + sourceLex[sourceKey]

	def project(self, projField, subset = {}):
		keys = self.select(subset)
		index = self.fields.index(projField)
		projection = {}
		for key in keys:
			projection[key[index]] = 1
		return(projection.keys())

	def read(self, fileName, encoding='utf-8', merge = True):
		fLex = codecs.open(fileName, 'rt', encoding)
		fields = fLex.readline().split()
		if fields[0][0] == u'\ufeff':		   # strip effin BOM
			fields[0] = fields[0][1:]

		if not self.fields:
			self.fields = fields
			self.nFields = len(fields)
		elif fields != self.fields:
			raise TypeError('Field mismatch between existing lexicon %s and update file %s\n' %(self.name, fileName))
				
		for line in fLex:
			fields = line.split()
			key = tuple(fields[:self.nFields])
			value = int(fields[-1])
			if merge: self[key] = self.get(key,0)+value
			else: self[key] = value
		fLex.close()

	def update(self, lex):
		if not self.fields: self.fields = lex.fields
		assert self.fields == lex.fields
		for key in lex: self[key] = self.get(key,0)+lex[key]		

	def write(self, fileName = '', encoding = 'utf-8'):
		if not fileName: fileName = self.name + '.lex'
		fLex = codecs.open(fileName, 'wt', encoding)
		for field in self.fields:
			fLex.write("%s\t" %field)
		fLex.write('\n')

		for entry in self:
			for field in entry:
				fLex.write("%s\t" %field)
			fLex.write("%d\n" %self[entry])
		fLex.close()
