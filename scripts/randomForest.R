# Sample Script using Random Forest benchmark

setwd("~/Kaggle/number/dataset/")

library(randomForest)

set.seed(0)

numTrain <- 22000
numTrees <- 25

# train <- read.csv("train.csv")
# test <- read.csv("test.csv")
# Fast way of reading a csv file
library(data.table)
train <- fread('train.csv', header = T, sep = ',')
test <- fread('test.csv', header = T, sep = ',')

# Get the training and test sets 
rows <- sample(1:nrow(train), 60)
labels <- train[rows, label]
train1 <- subset(train, select = -label)[rows, ]
test1 <- subset(train, select = label)[-rows, ]

# Random forest 
# The forest needs to be kept if xtest is being used
rf <- randomForest(train, labels, xtest=test, ntree=numTrees, keep.forest = TRUE)
predictions <- predict(rf, test)
predictT <- data.frame(ImageId=1:nrow(test), Label=round(predictions))
head(predictT)

write.csv(predictT, file = "submit.csv" , row.names = FALSE)


