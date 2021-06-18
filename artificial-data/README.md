# Hand-made artificial morphologies

> NB! In later research with randomly generated inputs, see each run's
> `*_input_table.txt` file for the artificial grammar and type frequencies of
> that run.

These datasets were created to explore how to generate homogeneous datasets
that vary in degree of implicative structure. They are relatively
small (5 classes and 5 morphosyntactic property sets) but vary in implicative
strength and should be homogeneous. Data_1 has strong implicative
structure (perfect predictability; conditional entropy = 0); data sets with
higher numbers have successively less strong implicative structure.

The first column 'typeFreq' is included to make these have the same structure
as files from earlier projects.  However, the type frequencies do not have
zipfian-like distributions here, so we/you can change that or write the
sampling distribution into the agent-based model.
