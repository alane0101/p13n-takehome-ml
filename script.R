# Set seed
set.seed(11235)

# Require packages

packages <- c("utils", "R.utils", "dplyr", "caret", "stats", "FactoMineR", "lubridate", "tidyr", "rlist", "e1071")
lapply(packages, library, character.only = TRUE)


# Unzip (call only works once, so it's commented out)

# untar("p13-takehome-ml.tar.gz", compressed = TRUE)

setwd("./p13-takehome-ml") # run in console


# Explore

list.files() # console

training <- read.csv("sampled_training", header = TRUE)

varnames <- names(training)
str(training) # console
summary(training) # console
anyNA(training) # console
head(training) # console


uniques <- sapply(as.list(varnames), function(varname) {
  length(unique(training[,varname]))
})

levels <- data.frame(varnames, uniques)

# NZV is unimpressed by 'device_id' and 'C16'  

nzv_training <- training[,-(nearZeroVar(training))]

# 'id' has the same number of unique entries as the number of observations, so it's not predictive and is probably an ID assigned by the ad server. 
# 'device_ip' is removed as it is also large and noisy. 

sub_training <- nzv_training[,-c(1,12)] # 'click' can stay



# Prep a version where we split timestamps into separate day of the week and time of day columns:

training_tod <- separate(sub_training, hour, into = c("Day", "Hour"), sep = -2) 
training_tod$Day <- strptime(training_tod$Day, format = "%y%m%d", tz = "GMT")
training_tod$Day <- wday(training_tod$Day, label = FALSE) # 1 = Mon; 7 = Sunday


training_tod2 <- separate(training[,-1], hour, into = c("Day", "Hour"), sep = -2)
training_tod2$Day <- strptime(training_tod2$Day, format = "%y%m%d", tz = "GMT")
training_tod2$Day <- wday(training_tod2$Day, label = FALSE) # 1 = Mon; 7 = Sunday


# Build models

model1 <- naiveBayes(factor(click) ~., data = training[,-1])

model2 <- naiveBayes(factor(click) ~., data = sub_training)

model3 <- naiveBayes(factor(click) ~., data = training_tod)

model4 <- naiveBayes(factor(click) ~., data = training_tod2)


# Ready testing data

testing <- read.csv("sampled_test", header = FALSE, col.names = varnames[-2])

sub_testing <- testing[,-c(1, 11, 12, 18)]

testing_tod <- separate(sub_testing, hour, into = c("Day", "Hour"), sep = -2)
testing_tod$Day <- strptime(testing_tod$Day, format = "%y%m%d", tz = "GMT")
testing_tod$Day <- wday(testing_tod$Day, label = FALSE)

testing_tod2 <- separate(testing[,-1], hour, into = c("Day", "Hour"), sep = -2)
testing_tod2$Day <- strptime(testing_tod2$Day, format = "%y%m%d", tz = "GMT")
testing_tod2$Day <- wday(testing_tod2$Day, label = FALSE)

# Predict

pred1 <- predict(model1, newdata = testing[,-1], type = "raw")

pred2 <- predict(model2, newdata = sub_testing, type = "raw")

pred3 <- predict(model3, newdata = testing_tod, type = "raw")

pred4 <- predict(model4, newdata = testing_tod2, type = "raw")


# Show the likelihood of a click for each entry in the test set, with a number between 0 and 1

outcomes1 <- data.frame(pred1[,2])
outcomes2 <- data.frame(pred2[,2])
outcomes3 <- data.frame(pred3[,2])
outcomes4 <- data.frame(pred4[,2])

# In-sample/misclassification error demonstration

ISE1 <- mean(predict(model1, newdata = training[,-c(1,2)], type = "class") != training$click)  # 0.5096941
ISE2 <- mean(predict(model2, newdata = training[,-c(1,2,12,13,19)], type = "class") != training$click)  # 0.3428065
ISE3 <- mean(predict(model3, newdata = training_tod[,-1], type = "class") != training$click)  #  0.3440762
ISE4 <- mean(predict(model4, newdata = training_tod2[,-1], type = "class") != training$click) # 0.5094641

# Export predictions

x <- data.frame(testing$id, outcomes2)
colnames(x) <- c("ad_id", "prob_click")
write.csv(x, file = "final_pred.csv", quote = FALSE, row.names = FALSE)
