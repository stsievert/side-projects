# This is an example R script for the InnoCentive Challenge located at:
#  https://www.innocentive.com/ar/challenge/9932794
#
# In order to generate a valid test result, your uploaded file 'main.R' must write out a file called 'predicted.out' containing 80 newline-delimited survival probabilities, in order of ascending patient ID (same order as in ValidationGenes.csv)


### LOAD REQUIRED PACKAGES ###
# Note: packages are automatically installed after upload
# Example using tourr
library(tourr)

# Added. -S.
library(stats)

### READ TRAINING DATA FROM CURRENT DIRECTORY ###
valgenes <- read.csv("ValidationGenes.csv", header=TRUE) # Be sure to include 'ValidationGenes.csv' in the uploaded archive!
length(valgenes$pat.id)

# Solve the system.
# phi=MxN measurements.
# x0= intial value of vector to be recovered. 
# T = sparsity basis (NxN: idenity matrix?)
# y = measurement matrix (Mx1)
 solve1TV(phi,y,T,x0,lamda=0.1)


### USING GENETIC VARIABLES, BUILD A MODEL WITH <= 10 PREDICTORS ###
# buildmodel <- function(n) {}


### USING THE MODEL, PREDICT THE PROBABILITY OF SURVIVAL FOR EACH INDIVIDUAL IN THE DATA ###
# predicted <- function(n) {}
predicted <- c(rnorm(80, 0.5, 0.1)) # As an example only, make a random prediction

### WRITE OUT THE PREDICTION ###
write.table(predicted, file="predicted.out", append=FALSE, quote=TRUE, sep=",", col.names=FALSE, row.names=FALSE, qmethod="escape")
