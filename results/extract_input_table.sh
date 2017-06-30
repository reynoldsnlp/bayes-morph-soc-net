#!/bin/bash

for input in 5.{1..23}
do
	pattern="*${input}.log"
	for each in $(ls ${pattern})
	do
		table=$(echo $each | sed 's/\.log/_input_table.txt/g')
		echo "generating ${table} ..."
		paste ../../artificial-data/id_column_24.tmp <(paste <(printf "typeFreq,$(grep lexeme_type_freq_list ${each} | cut -d [ -f 2 | sed 's/]//g' | sed 's/ //g')" | tr , \\n) "../../artificial-data/data_${input}.txt" | cut -f 1,3-) > ${table}
	done
done
