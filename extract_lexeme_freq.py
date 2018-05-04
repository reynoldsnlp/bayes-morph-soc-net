"""Load pickled model and extract lexeme frequency dist of last agent."""

import pickle
import sys

import test


filename = sys.argv[1]
print('Input file is "{}".'.format(filename))
filename = filename.rstrip('.pickle')
print('Output filename is "{}".'.format(filename))

print('Loading pickled model into memory... ({})'.format(filename))
with open(filename + '.pickle', 'rb') as p_file:
    model = pickle.load(p_file)
test.write_freq_dist(model, filename)
