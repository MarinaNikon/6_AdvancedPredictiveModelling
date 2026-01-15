#Assignment_Binary Logistic Regression using R
#Marina Nikon

#BACKGROUND:
#The Data for a Study of Risk Factors Associated with Low Infant Birth Weight. 
#Data were collected at Baystate Medical Center, Springfield, Massachusetts.


# Description of variables -
# LOW – Low Birth Weight (0 means Not low and 1 means low)
# AGE- Age of the Mother in Years
# LWT- Weight in Pounds at the Last Menstrual Period
# RACE- Race (1 = White, 2 = Black, 3 = Other)
# SMOKE- Smoking Status During Pregnancy (1 = Yes, 0 = No)
# PTL- History of Premature Labor (0 = None, 1 = One, etc.)
# HT- History of Hypertension (1 = Yes, 0 = No)
# UI- Presence of Uterine Irritability (1 = Yes, 0 = No)
# FTV- Number of Physician Visits During the First Trimester (0 = None, 1 = One, 2 = Two, etc.)

#Consider LOW as dependent variable and remaining variables listed above 
#as independent variables.


#Install the packages if required and call for the corresponding library
install.packages("ROCR") # ROC curve
library(gmodels) # for cross-tabulation
library(caret) # train-test, confusion matrix, k-fold validation
library(car) # to check Multicollinearity (VIF)
library(ROCR) # ROC curve


#QUESTIONS
#1. Import BIRTH WEIGHT data.

# Load the data
birthWeight<-read.csv("BIRTH_WEIGHT_csv.csv", header = TRUE)
head(birthWeight) # View first 6 rows
dim(birthWeight) # Check the dimension of the dataset
summary(birthWeight) #Summarizing data and checking for missing values
str(birthWeight) # Check the structure of the dataset
anyNA(birthWeight) # Check for missing values
birthWeight <- na.omit(birthWeight)

#Observations:
#There are no missing values, but there are some categorical variables
#that should be converted into "factor" variables

#Convert numerically coded categorical variables into factors
birthWeight$LOW<-as.factor(birthWeight$LOW)
birthWeight$RACE<-as.factor(birthWeight$RACE)
birthWeight$SMOKE<-as.factor(birthWeight$SMOKE)
birthWeight$PTL<-as.factor(birthWeight$PTL)
birthWeight$HT<-as.factor(birthWeight$HT)
birthWeight$UI<-as.factor(birthWeight$UI)
birthWeight$FTV<-as.ordered(birthWeight$FTV)

str(birthWeight)



#2. Cross tabulate dependent variable with each independent variable.

CrossTable(birthWeight$LOW) # simple cross table (LOW rate)
# Approximately 31% of infants were born with Low Weight

CrossTable(birthWeight$LOW, birthWeight$RACE, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Pearson's Chi-squared test 
#Chi^2 =  5.004813     d.f. =  2     p =  0.0818877

CrossTable(birthWeight$LOW, birthWeight$SMOKE, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Chi^2 =  4.923705     d.f. =  1     p =  0.02649064

CrossTable(birthWeight$LOW, birthWeight$PTL, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Chi^2 =  16.86387     d.f. =  3     p =  0.0007537928

CrossTable(birthWeight$LOW, birthWeight$HT, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Chi^2 =  4.387955     d.f. =  1     p =  0.0361937

CrossTable(birthWeight$LOW, birthWeight$UI, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Chi^2 =  6.24661     d.f. =  1     p =  0.01244312

CrossTable(birthWeight$LOW, birthWeight$FTV, prop.r = TRUE, prop.c = FALSE, chisq = T)
#Chi^2 =  5.986999     d.f. =  5     p =  0.307486

#Observations:
# AGE and LWT are numeric variables and have many unique levels.
#The data gets spread out, leading to smaller counts in individual cells,
#making the table harder to interpret visually.
# The AGE and LWT variables are grouped to make AGE and LWT tables more readable

# Create Age groups
birthWeight$AGE_GROUP <- cut(birthWeight$AGE, breaks = c(-Inf, 19, 29, Inf),
                             labels = c("<20", "20-29", "30+"))
CrossTable(birthWeight$LOW, birthWeight$AGE_GROUP, prop.r = TRUE, prop.c = FALSE, chisq = TRUE)
#Chi^2 =  4.660879     d.f. =  2     p =  0.09725298 

# Create LWT quartiles groups
summary(birthWeight$LWT)  # To see quartiles
quartiles <- quantile(birthWeight$LWT, probs = seq(0,1,0.25), na.rm = TRUE)
birthWeight$LWT_GROUP <- cut(birthWeight$LWT, breaks = quartiles, include.lowest=TRUE,
                labels = c("Q1 (80-110)", "Q2 (111-121)", "Q3 (122-140)", "Q4 (141-250)"))
CrossTable(birthWeight$LOW, birthWeight$LWT_GROUP, prop.r = TRUE, prop.c = FALSE, chisq = TRUE)
#Chi^2 =  8.822246     d.f. =  3     p =  0.03175001

#Observations:
#For RACE, FTV and AGE = p-value > 0.05 It could be suggested that RACE, AGE and the Number of
#Physician Visits During the First Trimester does not significantly affect a baby's birth weight'



# 3. Develop a model to predict if birth weight is low or not using the given variables.
# Logistic Regression Model
model<-glm(LOW~AGE + LWT + RACE + SMOKE + PTL + HT + UI + FTV,
           data=birthWeight,family="binomial")
summary(model)
#Observations:
# AIC: 220.17 # the lower, the better

# Significant predictors of low birth weight in this model are:
# LWT /Lower weight (p = 0.04675)
# RACE2 (p = 0.03964)
# PTL1 /one premature labor (p = 0.00119)
# HT1 / hypertension (p = 0.01655)
# UI1 / Uterine Irritability(p = 0.07508) slightly significant.

#Other predictors are not statistically significant

#NOTE:
# IN healthcare data, keeping all variables—including insignificant ones—can be beneficial
#for a comprehensive understanding, even if they are not statistically significant.
#Therefore, retaining these variables in the model allows for a more holistic analysis 
#that aligns with the complex and multifaceted nature of healthcare data.

# Check for multicollinearity using VIF
vif(model) # Use the third column: GVIF^(1/(2*Df))

#Observations:
#It is observed that no variable has high vif. Hence the problem of multicollinearity
#does not exist. Do not need to do re-modelling to remove multicollinearity


# 4. Generate three classification tables with cut-off values 0.4, 0.3 and 0.55.

# Split the data into Training (80%) and Testing (20%) datasets (Hold-out validation)

set.seed(123) #to use the same train-data
trainIndex <- createDataPartition(birthWeight$LOW, p = 0.8, list = FALSE)
head(trainIndex)
length(trainIndex)

traindata <- birthWeight[trainIndex, ]
testdata <- birthWeight[-trainIndex, ]

dim(traindata) #dimension of training set
dim(testdata) #dimension of testing set


#Binary Logistic Regression using glm function
#Train logistic regression model
birthWeight_model <- glm(LOW ~ AGE + LWT + RACE + SMOKE + PTL + HT + UI + FTV, 
                         data = traindata, family = "binomial")

summary(birthWeight_model)
 
#Observations:
# AIC: 172.01 # the lower, the better

# Significant predictors of low birth weight in this model are:
# LWT /Lower weight (p = 0.037159)
# PTL1 /one premature labor (p = 0.000337)
# HT1 / hypertension (p = 0.084274 ) slightly significant
# UI1 / Uterine Irritability(p = 0.027193)  

#Other predictors are not statistically significant

# Check for multicollinearity using VIF
vif(birthWeight_model) 
  
#Observations:
#It is observed that no variable has high vif. Hence the problem of multicollinearity
#does not exist. Do not need to do re-modelling to remove multicollinearity
  


# Prediction of train data with cut-off values 0.55
predictions_train <- predict(birthWeight_model, newdata = traindata, type = "response")
predicted_LOW_0.55_train <- ifelse(predictions_train > 0.55, 1, 0)
# Confusion Matrix train data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.55_train<-table(as.factor(predicted_LOW_0.55_train), traindata$LOW)
confusionMatrix_0.55_train
rownames(confusionMatrix_0.55_train) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.55_train) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.55_train)
#Observations:
#             Normal Weight LOW Weight Sum
#Normal Weight            96         23 119
#LOW Weight                8         25  33
#Sum                     104         48 152


# Prediction of test data with cut-off values 0.55
predictions_test <- predict(birthWeight_model, newdata = testdata, type = "response")
predicted_LOW_0.55_test <- ifelse(predictions_test > 0.55, 1, 0)
# Confusion Matrix for test data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.55_test<-table(as.factor(predicted_LOW_0.55_test), testdata$LOW)
confusionMatrix_0.55_test
rownames(confusionMatrix_0.55_test) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.55_test) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.55_test)
#               Normal Weight LOW Weight Sum
#Normal Weight            20          9  29
#LOW Weight                6          2   8
#Sum                      26         11  37


# Prediction of train data with cut-off values 0.4
predictions_train <- predict(birthWeight_model, newdata = traindata, type = "response")
predicted_LOW_0.4_train <- ifelse(predictions_train > 0.4, 1, 0)
# Confusion Matrix train data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.4_train<-table(as.factor(predicted_LOW_0.4_train), traindata$LOW)
confusionMatrix_0.4_train
rownames(confusionMatrix_0.4_train) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.4_train) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.4_train)
#Observations:
#               Normal Weight LOW Weight Sum
#Normal Weight            90         19 109
#LOW Weight               14         29  43
#Sum                     104         48 152


# Prediction of test data with cut-off values 0.4
predictions_test <- predict(birthWeight_model, newdata = testdata, type = "response")
predicted_LOW_0.4_test <- ifelse(predictions_test > 0.4, 1, 0)
# Confusion Matrix for test data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.4_test<-table(as.factor(predicted_LOW_0.4_test), testdata$LOW)
confusionMatrix_0.4_test
rownames(confusionMatrix_0.4_test) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.4_test) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.4_test)

#Observations:
#               Normal Weight LOW Weight Sum
#Normal Weight            18          8  26
#LOW Weight                8          3  11
#Sum                      26         11  37


# Prediction of train data with cut-off values 0.3
predictions_train <- predict(birthWeight_model, newdata = traindata, type = "response")
predicted_LOW_0.3_train <- ifelse(predictions_train > 0.3, 1, 0)
# Confusion Matrix train data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.3_train<-table(as.factor(predicted_LOW_0.3_train), traindata$LOW)
confusionMatrix_0.3_train
rownames(confusionMatrix_0.3_train) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.3_train) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.3_train)
#Observations:
#               Normal Weight LOW Weight Sum
#Normal Weight            82         11  93
#LOW Weight               22         37  59
#Sum                     104         48 152
            
            
# Prediction of test data with cut-off values 0.3
predictions_test <- predict(birthWeight_model, newdata = testdata, type = "response")
predicted_LOW_0.3_test <- ifelse(predictions_test > 0.3, 1, 0)
# Confusion Matrix for test data
# The classification table is also known as confusion matrix or error table!!!!
confusionMatrix_0.3_test<-table(as.factor(predicted_LOW_0.3_test), testdata$LOW)
confusionMatrix_0.3_test
rownames(confusionMatrix_0.3_test) <- c("Normal Weight", "LOW Weight")
colnames(confusionMatrix_0.3_test) <- c("Normal Weight", "LOW Weight")
addmargins(confusionMatrix_0.3_test)
#Observations:
#                Normal Weight LOW Weight Sum
#Normal Weight            14          6  20
#LOW Weight               12          5  17
#Sum                      26         11  37
  


#5. Calculate sensitivity, specificity and misclassification rate for all
#three tables above. What is the recommended cut-off value?
#  @ What are the ideal values of sensitivity and specificity for a model?
#  @ They have to be balanced, like they can't be one - 0%, another - 100%
#  @ Should be close to each other, the higher the sensitivity and specificity
#  @ are, the better

# Sensitivity and specificity for the training data with cut-off 0.55
confusionMatrix(as.factor(predicted_LOW_0.55_train), traindata$LOW)
#Observations:
# Accuracy : 0.7961 - means the model correctly predicts 80% of the time (good)
# Sensitivity : 0.9231 - the model correctly identifies 92% of normal birth weight (good)        
# Specificity : 0.5208 - the model correctly identifies 52% of low birth weight (low)       


# Sensitivity and specificity for the testing data with cut-off 0.55
confusionMatrix(as.factor(predicted_LOW_0.55_test), testdata$LOW)
#Observations:
# Accuracy : 0.5946 - means the model correctly predicts 59% of the time (low)
# Sensitivity : 0.7692 - the model correctly identifies 77% of normal birth weight (decent)         
# Specificity : 0.1818 - the model correctly identifies 18% of low birth weight (very poor)       


# Sensitivity and specificity for the training data with cut-off 0.4
confusionMatrix(as.factor(predicted_LOW_0.4_train), traindata$LOW)
#Observations:
# Accuracy : 0.7829 - means the model correctly predicts 78% of the time (good)
# Sensitivity : 0.8654 - the model correctly identifies 87% of normal birth weight (good)         
# Specificity : 0.6042 - the model correctly identifies 60% of low birth weight (better than 0.55 cut-off)        


# Sensitivity and specificity for the testing data with cut-off 0.4
confusionMatrix(as.factor(predicted_LOW_0.4_test), testdata$LOW)
#Observations:
# Accuracy : 0.5676 - means the model correctly predicts 57% of the time (low, worse than 0.55)
# Sensitivity : 0.6923 - the model correctly identifies 69% of normal birth weight (lower than 0.55)         
# Specificity : 0.2727 - the model correctly identifies 27% of low birth weight (low, but better then 0.55)       


# Sensitivity and specificity for the training data with cut-off 0.3
confusionMatrix(as.factor(predicted_LOW_0.3_train), traindata$LOW)
#Observations:
# Accuracy : 0.7829 - means the model correctly predicts 78% of the time (same as 0.4)
# Sensitivity : 0.7885 - the model correctly identifies 79% of normal birth weight (lower than 0.4 and 0.55)        
# Specificity : 0.7708 - the model correctly identifies 77% of low birth weight (better than 0.4 and 0.55)       


# Sensitivity and specificity for the testing data with cut-off 0.3
confusionMatrix(as.factor(predicted_LOW_0.3_test), testdata$LOW)
#Observations:
# Accuracy : 0.5135 - means the model correctly predicts 51% of the time (low)
# Sensitivity : 0.5385 - the model correctly identifies 54% of normal birth weight (low)         
# Specificity : 0.4545 - the model correctly identifies 45% of low birth weight (low)       



# Analyze model accuracy and misclassification rate 
#table(predicted_LOW_0.55_train, traindata$LOW)
#  Misclassification rate for train data with cut-off 0.55
misclassification_rate_0.55_train <- (confusionMatrix_0.55_train[1,2] + 
                                        confusionMatrix_0.55_train[2,1]) / sum(confusionMatrix_0.55_train)
misclassification_rate_0.55_train
# 0.2039474 (low)
#Observations:
# About 20.4% of the training data samples with 0.55 cut-off were misclassified by the model.
#It means the model is performing well.


#  Misclassification rate for test data with cut-off 0.55
misclassification_rate_0.55_test <- (confusionMatrix_0.55_test[1,2] + 
                                       confusionMatrix_0.55_test[2,1]) / sum(confusionMatrix_0.55_test)
misclassification_rate_0.55_test
# 0.4054054 - too high, indicating poor performance, about 41% is misclassified


#  Misclassification rate for train data with cut-off 0.4
misclassification_rate_0.4_train <- (confusionMatrix_0.4_train[1,2] + 
                                        confusionMatrix_0.4_train[2,1]) / sum(confusionMatrix_0.4_train)
misclassification_rate_0.4_train
# 0.2171053
#Observations:
# About 22% of the training data samples with 0.4 cut-off were misclassified by the model.
#It means the model is performing well.

misclassification_rate_0.4_test <- (confusionMatrix_0.4_test[1,2] + 
                                       confusionMatrix_0.4_test[2,1]) / sum(confusionMatrix_0.4_test)
misclassification_rate_0.4_test
# 0.4324324 -too high
#Observations:
# About 43% of the testing data samples with 0.4 cut-off were misclassified by the model.



misclassification_rate_0.3_train <- (confusionMatrix_0.3_train[1,2] + 
                                       confusionMatrix_0.3_train[2,1]) / sum(confusionMatrix_0.3_train)
misclassification_rate_0.3_train
# 0.2171053
#Observations:
# About 22% of the training data samples with 0.3 cut-off were misclassified by the model.
#It means the model is performing well.

misclassification_rate_0.3_test <- (confusionMatrix_0.3_test[1,2] + 
                                      confusionMatrix_0.3_test[2,1]) / sum(confusionMatrix_0.3_test)
misclassification_rate_0.3_test
# 0.4864865 - too high
#Observations:
# About 49% of the testing data samples with 0.3 cut-off were misclassified by the model.



#6. Obtain ROC curve and report area under curve. 

traindata$predprob<-fitted(birthWeight_model)
predtrain<-prediction(traindata$predprob,traindata$LOW)
perftrain<-performance(predtrain,"tpr","fpr")
plot(perftrain, col="blue", main="ROC Curve (Train Data)")
abline(0,1, col="green")

testdata$predprob<-predict(birthWeight_model,testdata,type='response')
predtest<-prediction(testdata$predprob,testdata$LOW)
perftest<-performance(predtest,"tpr","fpr")
plot(perftest, col="blue", main="ROC Curve (Test Data)")
abline(0,1, col="green")


#Checking area under the ROC curve
auctrain<-performance(predtrain,"auc")
auctrain@y.values
#Observations:
#0.8415465 - good
#Anything about 70 is a good auc value

auctest<-performance(predtest,"auc")
auctest@y.values
#Observations:
#0.520979 - poor

#Observations:
# Training performance is best with cut-off 0.55, it has highest sensitivity.
# Testing performance is very poor for all three cut-offs
# Misclassification rates are too high in all test data
# It could be suggested that none of this cut-offs is really good and
# another cut-off should to be tried.

