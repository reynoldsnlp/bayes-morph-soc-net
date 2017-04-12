#!/usr/bin/python
import numpy

consonants = [u'\u0431', u'\u0432', u'\u0433', u'\u0434', \
u'\u0436', u'\u0437', u'\u0439', u'\u043a', \
u'\u043b', u'\u043c', u'\u043d', u'\u043f', \
u'\u0440', u'\u0441', u'\u0442', u'\u0444', \
u'\u0445', u'\u0446', u'\u0447', u'\u0448', \
 u'\u0449', u'\u0436\u0434', u'\u0436\u0436', u'\u0436\u0437']
# b, v, g, d
# Z, z, j, k
# l, m, n, p
# r, s, t, f
# x, c, C, S,
# SC, dZ, ZZ, zZ

## feature-distance matrix
fdMatrix = numpy.array( [ \
	[0,1,1,1,3,2,3,2,3,1,2,1,3,3,2,2,3,4,3,4,3, 2, 2, 2], \
	[1,0,2,2,2,1,3,3,3,2,3,2,3,2,3,1,2,4,3,3,2, 2, 1, 1], \
	[1,2,0,1,3,2,3,1,3,2,2,2,3,3,2,3,2,4,3,4,3, 2, 2, 2], \
	[1,2,1,0,3,1,3,2,2,2,1,2,2,2,1,3,3,3,3,4,3, 2, 2, 2], \
	[3,2,3,3,0,2,3,4,4,4,4,4,4,3,4,3,3,3,3,1,2, 2, 1, 1], \
	[2,1,2,1,2,0,3,3,2,3,2,3,2,1,2,2,2,3,3,3,2, 2, 1, 1], \
	[3,3,3,3,3,3,0,4,2,2,2,4,2,4,4,4,4,5,3,4,3, 2, 2, 2], \
	[2,3,1,2,4,3,4,0,4,3,3,1,4,2,1,2,1,3,2,3,2, 3, 3, 3], \
	[3,3,3,2,4,2,2,4,0,2,1,4,1,3,3,4,4,4,4,5,4, 3, 3, 3], \
	[1,2,2,2,4,3,2,3,2,0,1,2,2,4,3,3,4,5,4,5,4, 3, 3, 3], \
	[2,3,2,1,4,2,2,3,1,1,0,3,1,3,2,4,4,4,4,5,4, 3, 3, 3], \
	[1,2,2,2,4,3,4,1,4,2,3,0,4,2,1,1,2,3,2,3,2, 3, 3, 3], \
	[3,3,3,2,4,2,2,4,1,2,1,4,0,3,3,4,4,4,4,5,4, 3, 3, 3], \
	[3,2,3,2,3,1,4,2,3,4,3,2,3,0,1,1,1,2,2,2,1, 2, 2, 2], \
	[2,3,2,1,4,2,4,1,3,3,2,1,3,1,0,2,2,2,3,3,2, 3, 3, 3], \
	[2,1,3,3,3,2,4,2,4,3,4,1,4,1,2,0,1,3,2,2,1, 3, 2, 2], \
	[3,2,2,3,3,2,4,1,4,4,4,2,4,1,2,1,0,3,2,2,1, 3, 2, 2], \
	[4,4,4,3,3,3,5,3,4,5,4,3,4,2,2,3,3,0,2,2,3, 3, 4, 4], \
	[3,3,3,3,3,3,3,2,4,4,4,2,4,2,3,2,2,2,0,2,1, 1, 2, 2], \
	[4,3,4,4,1,3,4,3,5,5,5,3,5,2,3,2,2,2,2,0,1, 3, 2, 2], \
	[3,2,3,3,2,2,3,2,4,4,4,2,4,1,2,1,1,3,1,1,0, 2, 1, 1], \
	[2,1,2,2,1,1,2,3,3,3,3,3,3,2,3,2,2,4,2,2,1, 0, 0, 1], \
	[2,1,2,2,1,1,2,3,3,3,3,3,3,2,3,2,2,4,2,2,1, 0, 0, 1], \
	[2,1,2,2,1,1,2,3,3,3,3,3,3,2,3,2,2,4,2,2,1, 1, 1, 0]], \
	float)

# default inflections -- not yet fitted for consonant mutation or reflexive verbs
infl = {'1s': u'\u0443', '2s': u'\u0438\u0448\u044c', '3s': u'\u0438\u0442', '1p': u'\u0438\u043c', '2p': u'\u0438\u0442\u0435', '3p': u'\u044f\u0442'}	

def stemConsonant(infinitive):
	if infinitive[-1] == u'\u044c': locStemC = -4	## default (ends in soft sign)
	else: locStemC = -6					## reflexive/passive verb (ends in c'a)
	c = infinitive[locStemC]

	## special case: double vowel signs ([oj], [ej], [aj], [uj]) --> j
	if c in [u'\u043e', u'\u0435', u'\u0430', u'\u0443']: c = u'\u0439'

	## special case: [ZZ], [zZ], [dZ]
	if c == [u'\u0436']:
		if infinitive[locStemC-1] in [u'\u0434', u'\u0436', u'\u0437']:
			c = infinitive[locStemC-1:locStemC]
	return(c)

def mpSim(lStr, rStr, consList = consonants, featDist = fdMatrix):
	lStemC, rStemC = stemConsonant(lStr), stemConsonant(rStr)
	try: dist = float(featDist[consList.index(lStemC), consList.index(rStemC)])
	except ValueError:
		print >> sys.stderr, 'mpSim exception for ', lStr, rStr
		return(0.0)
	return(max(1-dist/3.0,0.0))
