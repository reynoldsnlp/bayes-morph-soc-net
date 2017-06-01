for each in artificial-data/*.txt
do
	sh slurm.sh ${each}
done
