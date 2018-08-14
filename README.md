# bayes-morph-soc-net

Social network agent modeling with bayesian learning of artificial inflectional
morphologies. A description of the system can be found in...

Parker, Jeff, Robert Reynolds, and Andrea D. Sims. "A Bayesian Investigation of
Factors Shaping the Network Structure of Inflection Class Systems." Proceedings
of the Society for Computation in Linguistics 1, no. 1 (2018): 223-224.

## Installing dependencies

In order to install the necessary dependencies, run...


```bash
$ pip install -r requirements.txt
```

We also make use of some code in [rmalouf's
repository](https://github.com/rmalouf/morphology), copied as `entropy.py`.

## Running a model

The file `bmsn.py` contains all of the functions and classes to run a model.
The file `test.py` shows an example of how to implement a model--this is the
same file used to run models for the article cited above.
