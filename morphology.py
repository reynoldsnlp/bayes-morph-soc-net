"""Representation of morphological implicational structure."""


class Morphology:
    """Represent natural and artificial morphological paradigms."""

    def __init__(self, input_file):
        """Parse input file into Morphology object.

        Input is file object, tab-separated in the follow format.

        The first row contains headers, 'typeFreq' followed by MSPSs:
        typeFreq    MSPS1   MSPS2   MSPS3   etc.    ...

        Each following row represents an inflection class:
        457         a       a       i       etc.    ...
        12          a       i       i       etc.    ...
        ...
        ...
        """
        self.name = input_file.name
        file_lines = input_file.readlines()
        self.cols = [i.rstrip() for i in file_lines[0].split('\t')]
        self.MSPSs = self.cols[1:]
        self.infl_classes = []  # list of dicts
        for line in file_lines[1:]:
            infl_class = {}
            for i, datum in enumerate(line.split('\t')):
                try:
                    infl_class[self.cols[i]] = int(datum)
                except ValueError:
                    infl_class[self.cols[i]] = datum.rstrip()
            self.infl_classes.append(infl_class)
        self.msps_dict = {}
        for msps in self.MSPSs:
            self.msps_dict[msps] = {}
            for i_class in self.infl_classes:
                try:
                    self.msps_dict[msps][i_class[msps]] += i_class['typeFreq']
                except KeyError:
                    self.msps_dict[msps][i_class[msps]] = i_class['typeFreq']
        self.mnb = {}
        for target_msps in self.MSPSs:
            for given_msps in self.MSPSs:
                if target_msps != given_msps:
                    for given_ending in self.msps_dict[given_msps]:
                        tgg = (target_msps, given_msps, given_ending)
                        self.mnb[tgg] = self.mean_neighbor_behaviors(*tgg)

    def mean_neighbor_behaviors(self, target_msps, given_msps, given_ending):
        """Calculate type-freq-weighted prevalence of endings.

        Given an MSPS and its ending, return the probability of each
        ending for a particular MSPS, e.g...

        NomSg | (AccSg = -u) -> {ø: 0.0, a: 1.0, o: 0.0}

        Return dictionary of ending:probability pairs.
        """
        out_dict = {}
        cum_type_freq = 0
        for i in self.infl_classes:
            if i[given_msps] == given_ending:
                try:
                    out_dict[i[target_msps]] += i['typeFreq']
                except KeyError:
                    out_dict[i[target_msps]] = i['typeFreq']
                cum_type_freq += i['typeFreq']
        return {k: v/cum_type_freq for k, v in out_dict.items()}
