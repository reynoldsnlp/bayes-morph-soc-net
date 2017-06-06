"""Load pickled model and analyze data."""

import pickle
import sys

import test

filename = sys.argv[1]
if '/' in filename:
    filename = filename.split('/')[1]
print('Input file is {}.'.format(filename))
filename = filename.rstrip('.pickle')
print('Output filename is {}.'.format(filename))

print('Loading pickled model into memory... ({})'.format(filename))
with open('results/' + filename + '.pickle', 'rb') as p_file:
    model = pickle.load(p_file)

test.draw_boxplots(model, out_filename=filename)
