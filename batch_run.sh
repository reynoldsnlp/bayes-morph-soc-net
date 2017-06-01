for each in artificial-data/*.txt
do
	sbatch slurm.sh ${each}
done
