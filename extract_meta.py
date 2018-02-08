"""Print prior_weight for a bmsn model."""

import pickle
import sys

from bmsn import metalog

filename = sys.argv[1]
print(f'processing {filename}...', file=sys.stderr)
out_filename = filename.split('/')[-1].rstrip('.pickle')

with open(filename, 'rb') as p_file:
    print('    unpickling model...')
    model = pickle.load(p_file)
print('    printing metadata to models_meta.tsv...')
metalog(model, explicit_filename=out_filename)
