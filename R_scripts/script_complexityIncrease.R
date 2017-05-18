########################################################
######
######  script to look at relationship between (ir)regugularity and complexity
######	mostly taken from script_graphs.R from Andrea (used for Word Structure paper)
######
######
######
######
######
##########################################################

	
##functions needed -
	#doCalc.fnc
	#get_cond_ent.fnc(get_ent)
	
#testing each part of the script by running one line at a time
#input<-paste0(paste(getwd()),"/russian_87classes.txt")
#i=1
#j=1

#creating subdirectories for output; can also use showWarnings = FALSE to ignore error if directory already exists
dir.create(paste0(getwd(),"/2_subtract_classes"))
dir.create(paste0(getwd(),"/2_pairwise_numbers"))
dir.create(paste0(getwd(),"/2_output"))

#Manually setting whether doCalc.fnc will use typefrequencies to weight probabilities
weighted = F

## only needed if you want to preserve Cyrilic characters (none in the output currently)
#Sys.setlocale("LC_CTYPE", "Russian")

## note to self -- in R for windows, you must unselect misc > buffered output or none of the progress markers will print utnil the full for loop finishes

complexityIncrease.fnc = function(input){
	source("script_entropy_calc_v3.R")
	source("script_get_values_2.R")
	weighted = F
	for(i in 1:length(input)){
		file = read.delim(input[i], header=T, sep="\t", encoding="UTF-8")
		for(j in 1:nrow(file)){
#Remove one row (i.e. one inflection class), and calculate entropy over the remainder
			typeFreq_index<-grep("typeFreq", names(file))
			type_freq_dropped = file[j,typeFreq_index] 
			
			temp = file[,c(typeFreq_index:ncol(file))]
			data = temp[-c(j),]	
#Strip directory
			file_name = strsplit(input[i], ".*/")
			file_name = file_name[[1]][2]
#Output directory
			out_file = paste("drop", j, "_", file_name, sep="")
			out = paste0(getwd(),"/2_subtract_classes/", out_file)
#Write out individual inflection class data -- one class removed
			write.table(data, out, quote=F, sep="\t", row.names=F, col.names=T, fileEncoding = "UTF-8")
#Do entropy calculations -- calling script_entropy_calc.R
			doCalc.fnc(out, weighted)
#File that is the output of doCalc.fnc()
			get_ent = paste(getwd(),"/2_pairwise_numbers/", strsplit(out_file,".txt"), "_entropy_nonweighted_Rcalc.txt", sep="")
#Get average numbers (inflectional system entropy) for output of doCalc.fnc() (one dropped class)
			values = get_cond_ent.fnc(get_ent)
			values = cbind(type_freq_dropped, values)
#Write out results to a composite file
			if(j == 1){
				write.table(values, paste("2_output/", "drop_class_", file_name,  sep=""), quote=F, sep="\t", row.names=F, col.names=T, fileEncoding = "UTF-8") #write out column names
			} else {
				write.table(values, paste("2_output/", "drop_class_", file_name,  sep=""), quote=F, sep="\t", row.names=F, col.names=F, fileEncoding = "UTF-8", append=T) #don't write out column names and append
			}
		}
	}
}


##create the set of files to go through

input_files<-vector(length=0)
for (i in 1:12){
	input_files[i]<-paste0(paste(getwd()),"/data",i,".txt")
}

complexityIncrease.fnc(input_files)
#complexityIncrease.fnc(paste0(paste(getwd()),"/data1.txt"))
