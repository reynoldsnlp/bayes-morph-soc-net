##############################################################################
##############################################################################
####
#### script to get entropy, conditional entropy and mutual inforamation from output files
####
#### Jeff Parker, June 2015;  modified Jan 2015 for French and Icelandic data
####
#### modified for isolated traits of Russian data
####
##############################################################################
##############################################################################

#library(stringr)

#############################
####
####  reading in data from output files, creating data frames to generate graphs from
####
#############################

## function that reads in a output files and calculates the min, mean and max entropy values
get_cond_ent.fnc = function(filename){
	location<-paste0(getwd(),"/2_pairwise_numbers/")
	df <- read.delim(filename, sep="\t")
	
	ent_uncond<-aggregate(data=df, FUN=mean, entropy_A ~ msps_A+msps_B)
	ent<-mean(ent_uncond$entropy_A)
	ent_cond<-aggregate(data=df, FUN=mean, entropy_AgB ~ msps_A+msps_B)
	cond_ent<-mean(ent_cond$entropy_AgB)
	MI<- ent - cond_ent
	if (length(grep("nonweighted",filename)) == 1){
		weighted <- 0
		}
	else{ weighted <- 1
	}

	distribution<-cbind(filename, weighted, ent, cond_ent, MI)
	distribution<-as.data.frame(distribution, stringsAsFactors = FALSE)
	return(distribution)
}

## old, to make sure the script was getting the classes correctly
#for (a in 1:length(pairwise_files)){
#	filename<-pairwise_files[a]	
#	matches <- regmatches(filename, gregexpr("[0-9][0-9]?", filename))
#	classes<-as.numeric(unlist(matches))
#	print(classes)
#}
