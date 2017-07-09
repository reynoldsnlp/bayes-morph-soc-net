#!/bin/bash

for input in {4..5}.{1..23}
do
	pattern="*${input}.log"
	for each in $(ls ${pattern})
	do
		table=$(echo $each | sed 's/\.log/_input_table.txt/g')
		echo "generating ${table} ..."
		paste /Users/robertreynolds/research/bayes-morph-soc-net/artificial-data/id_column_24.tmp <(paste <(printf "typeFreq,$(grep lexeme_type_freq_list ${each} | cut -d [ -f 2 | sed 's/]//g' | sed 's/ //g')" | tr , \\n) "/Users/robertreynolds/research/bayes-morph-soc-net/artificial-data/data_${input}.txt" | cut -f 1,3-) > ${table}
	done
done
