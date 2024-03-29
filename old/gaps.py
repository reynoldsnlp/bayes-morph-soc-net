#!/usr/bin/python

"""
USAGE: python gaps.py outfile targinfl lexfile1 [lexfile2 [...]]

	targinfl is target inflection as specified in lex file, e.g. 1s
	lexfiles are lexicon files as input to/output by model

OVERVIEW -- Collects absolute lexeme frequencies and relative frequencies
	of target inflections -- N and k, respectively. For each lexeme,
	returns the desired absolute and relative frequencies, as well as
	a measure of "gappiness".

OUTPUTS (to stdout)

		freq1 freq2 ...	| %inf1 %inf2 ... | gap1 gap2 ...
	lexeme1
	lexeme2
	...

where indices on the rows indicate different lexemes, and indices on the
	the columns indicate different *input* files. The expected use of
	this feature is that the "trajectory" of a lexeme can be tracked
	over multiple generations.
""" 
## FOR DETAILS OF THE THEORY PLEASE READ THE FOLLOWING LONG COMMENT
## We assume that k is generated by an underlying binomial distribution with parameter p.
## Want to obtain Pr(p < theta | N,k). This value can be obtained using Bayes' rule with
##	a reasonable prior on Pr(p). We found that a flat prior is too willing to assign
##	gappiness to low-frequency items. A stronger prior is required. The most natural
##	prior to use would be one that matches the underlying lexical distribution, and
##	owing to the computational properties of integrating the binomial distribution,
##	a beta distribution is the most suitable choice. An eyeball match to the lexical
##	distribution is obtained using the hyperparameters alpha=2, beta=4. Thus:
##
##	Pr(p < theta | k,N) = INTEGRAL[0..theta] beta(2,8)*binom(x,k,N) dx /
##	                      INTEGRAL[0..1]     beta(2,8)*binom(x,k,N) dx
##			    = betainc(k+2, N+8, theta)
##
##	where betainc is the incomplete (regularized) beta integral
##	betainc(a,b,x) = beta[x](a,b)/beta(a,b) and
##	beta[x](a,b) = integral{0..x} t^(a-1) * (1-t)^(b-1) dt
##	and beta(a,b) = beta[1](a,b), normalizing to a pdf.

import sys, codecs
from scipy.special import betainc
from lexicon import Lexicon

## PARAMETERS
infl_1sg, alpha_1sg, beta_1sg, thresh_1sg = [u'1s'], 2, 8, .02
infl_3sg3pl, alpha_3sg3pl, beta_3sg3pl, thresh_3sg3pl = [u'3s', u'3p'], 5.5, 4.5, .98

outfile = sys.argv[1]
lexfiles = [sys.argv[i] for i in xrange(2,len(sys.argv))]
lexes = [Lexicon(lexfile) for lexfile in lexfiles]

nLexes = len(lexes)

## ACCUMULATE COUNTS
k1sg_Dic, k3sg3pl_Dic, NDic = {}, {}, {}
for iLex, lex in enumerate(lexes):
	iConcept, iFunction = lex.fields.index(u'CONCEPT'), lex.fields.index(u'FUNCTION')
	for key in lex:
		concept, function, count = key[iConcept], key[iFunction], lex[key]
		if function in infl_1sg:
			k1sg_Dic[(iLex,concept)] = k1sg_Dic.get((iLex,concept),0) + count
		if function in infl_3sg3pl:
			k3sg3pl_Dic[(iLex,concept)] = k3sg3pl_Dic.get((iLex,concept),0) + count
		NDic[(iLex,concept)] = NDic.get((iLex,concept),0) + count

## GET ROWS
biglex = Lexicon()
for lex in lexes: biglex.update(lex)
concepts = sorted(biglex.project(u'CONCEPT'))

## ## ## ## need to rewrite so it does filenames ## ## ## ##
fout = codecs.open(outfile, 'wt', 'utf-8')
header = ['lexeme']						## lexeme name
header += ['1sg_%s' %lexfile for lexfile in lexfiles]		## freq's of 1sg
header += ['3sg3pl_%s' %lexfile for lexfile in lexfiles]	## freq's of 3sg+3pl
header += ['freq_%s' %lexfile for lexfile in lexfiles]		## lexeme freq's
header += ['gappy_%s' %lexfile for lexfile in lexfiles]		## confidences that p[1s]<.02 ^ p[3]< .98
print >> fout, u'\t'.join(header)
def fancy(k, N, alpha, beta, thresh):
	if N == 0: return(0.0)
	return(betainc(k+alpha, N-k+beta, thresh))

for concept in concepts:
	k1sg_List = [k1sg_Dic.get((iLex,concept),0) for iLex in range(nLexes)]
	k3sg3pl_List = [k3sg3pl_Dic.get((iLex,concept),0) for iLex in range(nLexes)]
	NList = [NDic.get((iLex,concept),0) for iLex in range(nLexes)]
	gapList = []
	for iLex in range(nLexes):
		gappy = fancy(k1sg_List[iLex], NList[iLex], alpha_1sg, beta_1sg, thresh_1sg)
		personal = fancy(k3sg3pl_List[iLex], NList[iLex], alpha_3sg3pl, beta_3sg3pl, thresh_3sg3pl)
		gapList.append(gappy*personal)
	print >> fout, concept + '\t' + u'\t'.join([str(x) for x in k1sg_List+k3sg3pl_List+NList+gapList])
fout.close()

## python gaps.py mp1.txt russkii.lex mp1/seed_output.lex mp1/gen0_output.lex mp1/gen1_output.lex mp1/gen2_output.lex mp1/gen3_output.lex mp1/gen4_output.lex mp1/gen5_output.lex mp1/gen6_output.lex mp1/gen7_output.lex mp1/gen8_output.lex mp1/gen9_output.lex
