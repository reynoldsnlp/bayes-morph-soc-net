#!/bin/bash

while [ 1 ]
do
	for dist in flat zipf
	do
		for weight in proportion_of_lex none
		do
			for each in 01 02 03 04 05 06 08 10
			do
				nice python3 test.py artificial-data/data_6.${each}.txt 60000 ${weight} ${dist}
			done
		done
	done
done
