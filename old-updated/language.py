#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
"""Define language."""

import numpy
import sys

CONSONANTS = ['б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р',
              'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'жд', 'жж', 'жз']
#              b,   v,   g,   d,   Z,   z,   j,   k,   l,   m,   n,   p,   r,
#              s,   t,   f,   x,   c,   C,   S,   SC,  Zd,   ZZ,   Zz

# feature-distance matrix
fdMatrix = numpy.array([
    [0, 1, 1, 1, 3, 2, 3, 2, 3, 1, 2, 1, 3, 3, 2, 2, 3, 4, 3, 4, 3, 2, 2, 2],
    [1, 0, 2, 2, 2, 1, 3, 3, 3, 2, 3, 2, 3, 2, 3, 1, 2, 4, 3, 3, 2, 2, 1, 1],
    [1, 2, 0, 1, 3, 2, 3, 1, 3, 2, 2, 2, 3, 3, 2, 3, 2, 4, 3, 4, 3, 2, 2, 2],
    [1, 2, 1, 0, 3, 1, 3, 2, 2, 2, 1, 2, 2, 2, 1, 3, 3, 3, 3, 4, 3, 2, 2, 2],
    [3, 2, 3, 3, 0, 2, 3, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 1, 2, 2, 1, 1],
    [2, 1, 2, 1, 2, 0, 3, 3, 2, 3, 2, 3, 2, 1, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1],
    [3, 3, 3, 3, 3, 3, 0, 4, 2, 2, 2, 4, 2, 4, 4, 4, 4, 5, 3, 4, 3, 2, 2, 2],
    [2, 3, 1, 2, 4, 3, 4, 0, 4, 3, 3, 1, 4, 2, 1, 2, 1, 3, 2, 3, 2, 3, 3, 3],
    [3, 3, 3, 2, 4, 2, 2, 4, 0, 2, 1, 4, 1, 3, 3, 4, 4, 4, 4, 5, 4, 3, 3, 3],
    [1, 2, 2, 2, 4, 3, 2, 3, 2, 0, 1, 2, 2, 4, 3, 3, 4, 5, 4, 5, 4, 3, 3, 3],
    [2, 3, 2, 1, 4, 2, 2, 3, 1, 1, 0, 3, 1, 3, 2, 4, 4, 4, 4, 5, 4, 3, 3, 3],
    [1, 2, 2, 2, 4, 3, 4, 1, 4, 2, 3, 0, 4, 2, 1, 1, 2, 3, 2, 3, 2, 3, 3, 3],
    [3, 3, 3, 2, 4, 2, 2, 4, 1, 2, 1, 4, 0, 3, 3, 4, 4, 4, 4, 5, 4, 3, 3, 3],
    [3, 2, 3, 2, 3, 1, 4, 2, 3, 4, 3, 2, 3, 0, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2],
    [2, 3, 2, 1, 4, 2, 4, 1, 3, 3, 2, 1, 3, 1, 0, 2, 2, 2, 3, 3, 2, 3, 3, 3],
    [2, 1, 3, 3, 3, 2, 4, 2, 4, 3, 4, 1, 4, 1, 2, 0, 1, 3, 2, 2, 1, 3, 2, 2],
    [3, 2, 2, 3, 3, 2, 4, 1, 4, 4, 4, 2, 4, 1, 2, 1, 0, 3, 2, 2, 1, 3, 2, 2],
    [4, 4, 4, 3, 3, 3, 5, 3, 4, 5, 4, 3, 4, 2, 2, 3, 3, 0, 2, 2, 3, 3, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 2, 4, 2, 3, 2, 2, 2, 0, 2, 1, 1, 2, 2],
    [4, 3, 4, 4, 1, 3, 4, 3, 5, 5, 5, 3, 5, 2, 3, 2, 2, 2, 2, 0, 1, 3, 2, 2],
    [3, 2, 3, 3, 2, 2, 3, 2, 4, 4, 4, 2, 4, 1, 2, 1, 1, 3, 1, 1, 0, 2, 1, 1],
    [2, 1, 2, 2, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 4, 2, 2, 1, 0, 0, 1],
    [2, 1, 2, 2, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 4, 2, 2, 1, 0, 0, 1],
    [2, 1, 2, 2, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 4, 2, 2, 1, 1, 1, 0]
    ], float)

# default inflections--not yet fitted for consonant mutation or reflexive verbs
infl = {'1s': 'у', '2s': 'ишь', '3s': 'ит', '1p': 'им', '2p': 'ите', '3p': 'ят'}


def stemConsonant(infinitive):
    """Return final stem consonant(s) of an infinitive."""
    if infinitive[-1] == 'ь':
        locStemC = -2            # default (ends in soft sign)
    else:
        locStemC = -3          # reflexive/passive verb (ends in c'a)
    # TODO(RJR): what about ти?
    # TODO(RJR): Should these be -2 and -3 in python3 (unicode)?
    c = infinitive[locStemC]

    # special case: double vowel signs ([oj], [ej], [aj], [uj]) --> j
    if c in ['о', 'е', 'а', 'у']:
        c = 'й'

    # special case: [ZZ], [zZ], [dZ]
    if c == ['ж']:
        if infinitive[locStemC-1] in ['д', 'ж', 'з']:
            c = infinitive[locStemC-1:locStemC]
    return c


def mpSim(lStr, rStr, consList=CONSONANTS, featDist=fdMatrix):
    # TODO(RJR) document this function.
    lStemC, rStemC = stemConsonant(lStr), stemConsonant(rStr)
    try:
        dist = float(featDist[consList.index(lStemC), consList.index(rStemC)])
    except ValueError:
        print('mpSim exception for ', lStr, rStr, file=sys.stderr)
        return 0.0
    return max(1-dist/3.0, 0.0)
