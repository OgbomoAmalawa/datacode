##### Cross Validation Demo: Classification using K Nearest Neighbors --------------------

## Example: Classifying Cancer Samples ----
# The data can be downloaded from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# Some parts of the following codes from "B. Lantz, Machine learning with R, PACKT, 2015, Ch3"

# Step 1: Import the CSV format data "wisc_bc_data.csv"
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = TRUE)

# Step 2: Explore the data, e.g. examine the structure of the wbcd data frame, normalisation, and so on
str(wbcd)

# drop the id featuren which is useless in this case
wbcd <- wbcd[-1]

# table of diagnosis (We ignore data balancing process)
table(wbcd$diagnosis)

# recode diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("0", "1"))

# table or proportions with more informative labels
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# summarize three numeric features
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# test normalization function - result should be identical
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))

# normalize the wbcd data
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
wbcd_n$diagnosis <- wbcd$diagnosis

# confirm that normalization worked
summary(wbcd_n$area_mean)

# Step 3: Split the data into developing subset (including training and validation subset) and testing subset
# pick up Malignant subset and Benign subset
MalignantSubset <- wbcd_n[(wbcd_n$diagnosis == 1), ]
BenignSubset <- wbcd_n[(wbcd_n$diagnosis == 0), ]

# create developing subset (80%) and testing subset (20%) in terms of Malignant and Benign respectively
smp1 <- floor(nrow(MalignantSubset)*4/5)
smp0 <- floor(nrow(BenignSubset)*4/5)

set.seed(1)
# Randomly generate smp1 indices for Malignant subset
idx1 <- sample(seq_len(nrow(MalignantSubset)), size = smp1) 
Malignant_deve <- MalignantSubset[idx1, ]
Malignant_test <- MalignantSubset[-idx1, ]

set.seed(0)
# Randomly generate smp0 indices for Benign subset
idx0 <- sample(seq_len(nrow(BenignSubset)), size = smp0) 
Benign_deve <- BenignSubset[idx0, ]
Benign_test <- BenignSubset[-idx0, ]

# Combine Malignant_test and Benign_test by rows as TestingSubset, which will be used to test the validated classifier
TestingSubset <- rbind(Malignant_test, Benign_test)

# Set 9/10 of the developing subset to be training subset
# the remaining 1/10 of the developing subset to be validation subset
fold <- 10 # 10-fold cross validation
smp1 <- floor(nrow(Malignant_deve)*9/10) 
smp0 <- floor(nrow(Benign_deve)*9/10) 
# Declare the 'train_ind' and 'corr' variables before using them
MalignantTrain_ind <- matrix(NA, nrow = fold, ncol = smp1) 
BenignTrain_ind <- matrix(NA, nrow = fold, ncol = smp0) 

#-Randomly sampling training indices ('train_ind[i,]') for each fold from developing subset
# This is an example to split data into 10 folds. But you can use other splitting approach
for (i in 1:fold){ 
  set.seed(i) # to guarantee repeatable results
  MalignantTrain_ind[i,] <- sample(seq_len(nrow(Malignant_deve)), size = smp1)
  
  set.seed(i*100)
  BenignTrain_ind[i,] <- sample(seq_len(nrow(Benign_deve)), size = smp0)
}

## Step 4: Train a model/classifier on the training subset and valide the classifier on validation seubset by 'fold'-fold cross validation----

# load the "class" library where the knn function is going to be used
library(class) 
# load the "descr" library where the CrossTable function is going to be used
library(descr)
N <- 20 # suppose we are trying 1~20 nearest neighbours
BA <- matrix(NA, nrow = 1, ncol = fold) # BA: Balanced accuracy for 'fold' folds in terms of a specific k
BA_K <- matrix(NA, nrow = N, ncol = 1)  # BA_K: Store average BA of 'fold' folds for each k

for (k in 1:N)
{
  for (i in 1:fold)
  {
    TrainSubset <- rbind(Malignant_deve[MalignantTrain_ind[i,],], Benign_deve[BenignTrain_ind[i,],])
    ValidSubset <- rbind(Malignant_deve[-MalignantTrain_ind[i,],], Benign_deve[-BenignTrain_ind[i,],])
    # Randomly reorder the elements of the training subset
    set.seed(i*k)
    TrainSubset <- TrainSubset[sample(nrow(TrainSubset)),]
    # Randomly reorder the elements of the validation subset
    set.seed(i*k+100)
    ValidSubset <- ValidSubset[sample(nrow(ValidSubset)),]
    # Train the knn classifier with different k
    valid_pred <- knn(train = TrainSubset[,1:30], test = ValidSubset[,1:30], cl = TrainSubset[,31], k)
    # Get the confusion matrix table
    CT <- CrossTable(x = ValidSubset[,31], y = valid_pred)
    # Calculate balanced accuracy (BA) for each fold
    BA[i] <- (CT$t[1,1]/(CT$t[1,1]+CT$t[1,2]) + CT$t[2,2]/(CT$t[2, 1]+CT$t[2,2]))/2
  }
  # Calculate the mean of BA for each k
  BA_K[k] <- mean(BA)
}

# Find the maximum balanced accuracy with the corresponding k, max_k
max_k <- which.max(BA_K) 

# Step 5: Test the classifer with trained k (max_k)
DeveSubset <- rbind(Malignant_deve, Benign_deve) # combining all developing sets
# Randomly reorder the elements of the developing subset
set.seed(12345)
DeveSubset <- DeveSubset[sample(nrow(DeveSubset)),]

deve_pred <- knn(train = DeveSubset[,1:30], test = TestingSubset[,1:30], cl = DeveSubset[,31], max_k)

CT <- CrossTable(x = TestingSubset[,31], y = deve_pred)
BA_Test <- (CT$t[1,1]/(CT$t[1,1]+CT$t[1,2]) + CT$t[2,2]/(CT$t[2, 1]+CT$t[2,2]))/2
