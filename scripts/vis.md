---
title: "V1"
author: "Yiyao Liu"
date: "December 31, 2015"
output: html_document
---
```{r, echo=FALSE, warning=FALSE}
# Sample Script using Random Forest benchmark

setwd("~/Desktop/R_Python")
library(randomForest)
library(data.table)
library(caret)
library(ROCR)

# Read data
train <- fread('train.csv', header = T, sep = ',')
test <- fread('test.csv', header = T, sep = ',')

# k-fold cross-validation
set.seed(2016)
k <- 0.3
rows <- sample(1:nrow(train), nrow(train)*k)
new_train <- train[-rows,]
new_test <- train[rows,]
train_labels <- as.factor(new_train[, label])
test_labels <- as.factor(new_test[, label])

# Random forest parmater
#numTrees <- 10
#n_trees <- c(1, 5, 10)
n_trees <- c(3, 10, 20, 40)
#n_trees <- c(10, 15, 25, 40, 70)

#function for each tree, output: a confusion matrix plot and a prediction rate
rate_cal <- function(numTrees)  
{
  rf <- randomForest(new_train, train_labels, ntree=numTrees, keep.forest = TRUE)
  #predictions <- predict(rf, new_test, type = 'prob')
  predictions2 <- predict(rf, new_test)
  #summary(predictions)
  confusion <-  table(test_labels, predictions2)
  rate <- sum(diag(confusion))/sum(confusion)
  confusion <-  as.data.frame(table(test_labels, predictions2))
  names(confusion) = c("Actual","Predicted","Freq")
  
  #render plot
  # we use three different layers
  # first we draw tiles and fill color based on percentage of test cases
  tile <- ggplot() +
    geom_tile(aes(x=Actual, y=Predicted,fill=Freq),data=confusion, color="black",size=0.1) +
    labs(x="Actual",y="Predicted")
  tile = tile + 
    geom_text(aes(x=Actual,y=Predicted, label=sprintf("%.1f", Freq)),data=confusion, size=3, colour="black") +
    scale_fill_gradient(low="white",high="red")
  
  # lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border
  tile = tile + 
    geom_tile(aes(x=Actual,y=Predicted),data=subset(confusion, as.character(Actual)==as.character(Predicted)), color="black",size=0.3, fill="black", alpha=0) 
  tile <- tile +
    labs(title = paste0("Confusion Matrix for ", numTrees, " Trees"))
  
  #return render and prediction rate
  return(list(tile, rate))
}

#apply the function
results <- lapply(n_trees, rate_cal)

#print the heat plot for each tree
for (i in 1:length(results)) {print(results[[i]][[1]])}

#extract the prediction rate
results_2 <- vector()
for (i in 1:length(results)) {results_2 <- rbind(results_2, unlist(results[[i]][[2]]))}
results_2 <- as.data.frame(cbind(results_2, n_trees))
names(results_2)[1] <- "Accuracy_Rate"

#plot the prediction rate against #tree
tile2 <- ggplot(results_2, aes(x = n_trees, y = Accuracy_Rate)) + geom_point(color="blue") + geom_line() 
tile2 <- tile2 + scale_x_continuous(breaks = results_2$n_trees) +
  labs(x="Number of Trees",y="Prediction Accuracy Rate of CV Test Data")
tile2 
```
