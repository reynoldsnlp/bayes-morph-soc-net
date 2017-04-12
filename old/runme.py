#!/usr/bin/python

import sys, os
from agent import Agent
from lexicon import Lexicon
from agentNetwork import AgentNetwork, connect
import language

from time import clock

################ OPTION PARSING AND DEFAULTS ################
##  Parses any command-line options.
##  Mainly this will just assign defaults.
#############################################################

from optparse import OptionParser, OptionGroup
parser = OptionParser(usage = "usage: %prog [options] inputlexiconfile")

## SOCIAL PARAMETERS
socialgroup = OptionGroup(parser, "SOCIAL PARAMETERS")
socialgroup.add_option("-g", "--generations", dest="nGens", type="int", help="number of generations (default=%default)", default=10)
socialgroup.add_option("-p", "--population", dest="nAgents", type="int", help="number of agents in a generation (default=%default)", default=50)
socialgroup.add_option("-k", "--neighbors", dest="avgNbrs", type="int", help="number of neighbors (default=%default)", default=10)
socialgroup.add_option("-l", "--lifetime", dest="lifetime", type="int", help="number of generations an agent survives (default=%default)", default=1)
parser.add_option_group(socialgroup)

## LEARNING AND PRODUCTION PARAMETERS
learngroup = OptionGroup(parser, "LEARNING AND PRODUCTION PARAMETERS")
learngroup.add_option("-b", "--beta", dest="beta", type="float", help="Strength of prior, relative to 1 observed form (default=%default)", default=1.0)
learngroup.add_option("-K", "--kappa", dest="kappa", type="int", help="Size of lexical neighborhood for prior (default=%default)", default=30)
learngroup.add_option("-u", "--utterances", dest="nUtts", type="int", help="Number of utterances in output cycle (default=%default)", default=1000000)
learngroup.add_option("-m", "--mpsim", dest="mpSim", action="store_true", help="Weight by morphophonological similarity (default=%default)", default=False)
parser.add_option_group(learngroup)

## INPUT/OUTPUT OPTIONS
iogroup = OptionGroup(parser, "INPUT/OUTPUT OPTIONS")
iogroup.add_option("-o", "--outdir", dest="outputDir", type="string", help="output directory (default=%default)", default=os.path.join(os.getcwd(),"output"))
iogroup.add_option("-a", dest="agentOutput", action="store_true", help="write individual output files (default=%default)", default=False)
parser.add_option_group(iogroup)

## DEBUGGING OPTIONS
debuggroup = OptionGroup(parser, "DEBUGGING")
debuggroup.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Extensive debugging commentary (default=%default)", default=False)
parser.add_option_group(debuggroup)

options, args = parser.parse_args()

try:
	assert len(args) == 1
	options.inputlexicon = args[0]
except AssertionError:
	print >> sys.stderr, "\nERROR: Something is wrong with the (obligatory) input lexicon file argument.\n"
	parser.print_help()
	sys.exit(1)

############################ MAIN ############################

if options.mpSim: similarity = language.mpSim
else: similarity = lambda x,y: 1.0
agentAttributes = {'beta': options.beta, 'kappa': options.kappa, 'simFunction': similarity}

seeder = Agent(agentAttributes)
seeder.lexicon = Lexicon(fileName = options.inputlexicon)
if options.verbose: print >> sys.stderr, '(%s) Input lexicon read' %str(clock())
seeder.learn()

output = [seeder.talk(options.nUtts) for i in xrange(options.nAgents)]
if options.verbose: print >> sys.stderr, '(%s) Initial generation\'s input sampled' %str(clock())

meta = Lexicon()
for lex in output: meta.update(lex)
meta.write(os.path.join(options.outputDir, 'seed_output.lex'))
if options.verbose: print >> sys.stderr, '(%s) Seed recorded' %str(clock())

community = AgentNetwork()
community.birth(options.nAgents, Agent, agentParams = agentAttributes)
community.adjList = connect(options.nAgents, method = 'poisson', type = list, connectParams = {'k': options.avgNbrs})
if options.verbose: print >> sys.stderr, '(%s) Initial generation birthed and connected' %str(clock())

for iGen in xrange(options.nGens):
	community.killRank(1)
	if options.verbose: print >> sys.stderr, '(%s) Generation %d died off' %(str(clock()), iGen-1)

	for iAgent, agent in enumerate(community.getRank(0)):
		for jTeacher in community.adjList[iAgent]: agent.listen(output[jTeacher])
	if options.verbose: print >> sys.stderr, '(%s) Generation %d listened' %(str(clock()), iGen)

	for iAgent, agent in enumerate(community.getRank(0)):
		agent.learn()
	if options.verbose: print >> sys.stderr, '(%s) Generation %d learned' %(str(clock()), iGen)

	community.promote()
	output = [agent.talk(options.nUtts) for agent in community.getRank(1)]
	if options.verbose: print >> sys.stderr, '(%s) Generation %d talked' %(str(clock()), iGen)

	meta = Lexicon()
	for lex in output: meta.update(lex)
	meta.write(os.path.join(options.outputDir, 'gen%d_output.lex' %iGen))
	if options.verbose: print >> sys.stderr, '(%s) Generation %d\'s output recorded' %(str(clock()), iGen)

	community.birth(options.nAgents, Agent, agentParams = {'beta': options.beta, 'kappa': options.kappa, 'simFunction': similarity})
	community.adjList = connect(options.nAgents, method = 'poisson', type = list, connectParams = {'k': options.avgNbrs})
	if options.verbose: print >> sys.stderr, '(%s) Generation %d birthed and connected' %(str(clock()), iGen+1)
