#loading the dataset
library(readr)
bank_full <- read_delim("E:/pooja/DS/ExcelR/R/Assignments/bank-full.csv", 
                        ";", escape_double = FALSE, trim_ws = TRUE)
View(bank_full)

#checking for null values
is.null(bank_full)

#Data pre-processing 
#Handling categorical variable

library(dummy)
library(sjmisc)
library(dplyr)

bank.dummy <- dummy_cols(bank_full)
bank_full$y <- ifelse(bank_full$y == "yes", 1, 0)
bank.dummy <- bank.dummy[,-c(2,3,4,5,7,8,9,11,16)]

bank.dummy <- sapply(bank.dummy, as.numeric)
bank.dummy <- as.data.frame(bank.dummy)

bank.dummy2 <- subset(bank.dummy, select = -c(y))
bank.dummy2 <- scale(bank.dummy2)

bank.dummy2 <- cbind(bank.dummy2, bank_full$y)

bank.dummy2 <- as.data.frame(bank.dummy2)

names(bank.dummy2)[names(bank.dummy2) == "V52"] <- "y"

bank.dummy2 <- sapply(bank.dummy2, as.numeric)

#EDA
hist(bank.dummy2$y)
boxplot(bank.dummy2$y)
cor(bank.dummy2)

#splitting data into training and testing data
library(caTools)
split <- sample.split(bank.dummy2$y, SplitRatio = 0.8)
bank_training_data <- subset(bank.dummy2, split == "TRUE")
bank_testing_data <- subset(bank.dummy2, split == "FALSE")

#fitting the model

bank_logit <- glm(y ~ ., data = bank_training_data, family = "binomial")
summary(bank_logit)

library(MASS)
library(car)

stepAIC(bank_logit)

exp(coef(bank_logit))

bank_prob <- predict(bank_logit, data= bank_testing_data, type = "response")
bank_prob

#predicting accuracy of model
confusion_matrix <- table(bank_prob > 0.5, bank_training_data$y)
confusion_matrix

accuracy.bank <- sum(diag(confusion_matrix)/sum(confusion_matrix))
accuracy.bank   #0.902292

specificity_bank <- confusion_matrix[1,2]/sum(confusion_matrix[1,2],confusion_matrix[2,2])
specificity_bank    #0.6497282

sensitivity_bank<- confusion_matrix[1,1]/sum(confusion_matrix[1,1],confusion_matrix[2,1])
sensitivity_bank   #0.9754211

plot(sensitivity_bank, specificity_bank)

library(ROCR)
ROCpred_bank <- prediction(bank_prob, bank_training_data$y)
ROCRPerf <- performance(ROCpred_bank, "tpr","fpr")

plot(ROCRPerf, colorize=TRUE, print.cutoffs.at = seq(0.1,by=0.1))
