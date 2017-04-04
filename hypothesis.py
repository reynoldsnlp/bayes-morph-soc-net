"""Hypothesis space for Bayesian learning model."""

from os import listdir
from os.path import isfile
from os.path import join

import pandas as pd

LG_DATA_DIR = 'language-data'

filenames = [f for f in listdir(LG_DATA_DIR) if isfile(join(LG_DATA_DIR, f))]
print(filenames)

languages = []
for filename in filenames:
    print('Importing {}...'.format(filename))
    df = pd.DataFrame.from_csv(LG_DATA_DIR+'/'+filename, sep='\t', header=0)
    print(df)
    languages.append(df)

print(languages)
hypotheses = []
