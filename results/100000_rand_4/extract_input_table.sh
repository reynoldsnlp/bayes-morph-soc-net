#!/bin/bash

# This script takes one argument, the numeric part of data_x.y.txt
# Ex: $ sh extract_input_table.sh 4.14

for each in "*"$1".log"; do table=$(echo $each | sed 's/\.log/_input_table.txt/g'); paste ../../artificial-data/id_column_24.tmp <(paste <(printf "typeFreq,$(grep lexeme_type_freq_list ${each} | cut -d [ -f 2 | sed 's/]//g' | sed 's/ //g')" | tr , \\n) "../../artificial-data/data_"$1".txt" | cut -f 1,3-) > ${table}; done

