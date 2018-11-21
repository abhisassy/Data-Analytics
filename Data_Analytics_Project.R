#Loading Packages 
library('ggplot2')      #visualization
library('ggthemes')     #visualization
library('scales')       #visualization
library('dplyr')        #data imputation
library('mice')         #data imputation
library('randomForest') #classification algorithm
library('InformationValue')
library('caTools')      #confussion matrix
library('ROCR')         #ROC curve 

#the training data
train_data = read.csv('Data/train.csv', stringsAsFactors = F)
#the test data
test_data  = read.csv('Data/test.csv', stringsAsFactors = F) 

#understanding the proportion of suvivors in training data set 
#checking bias towards non survivors
table(train_data$Survived)
head(train_data$Survived)
ggplot(train_data, aes(x=as.factor(train_data$Survived), fill=as.factor(train_data$Survived))) + geom_bar() + scale_fill_manual(values = c("red","green"))

#binding training and test data 
full =  bind_rows(train_data , test_data)


#TAKING CARE OF MISSING DATA

#1
#Under Embarkment two people have missing data
#We calculate the distribution of the fare paid amoung the three different classes , in the 3 different Embarkment points
#and try to find which median value is closest to $80 which is what they paid

# Get rid of our missing passenger IDs
embark_fare = full %>% 
  filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

#from the graph it is evedent that the embarkment should be C
full$Embarked[c(62, 830)] = 'C'; 


#2
#NA Farevalue on row 1044
full[1044,]
#3rd class passenger , embarked at S
#lets find the distribution of fare prices for all passengers with these criteria
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format())

#from given plot we can safely replace NA with the median value 
full$Fare[1044] = median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)


#3
#Misisng Age values 
sum(is.na(full$Age))#number of missing values in age

#Here we use predictive anaysis to fill in the missing values 
#for this we use the 'mice' package functionality
#which basically uses a regression model based on the remaining variables to predict the missing values
#linear regression for numerical data & logistic for categorical

factor_vars       = c ("Pclass", "Sex", "Embarked");
full[factor_vars] = lapply(full[factor_vars], function(x) as.factor(x))
set.seed(129)
mice_mod    = mice(full[, !names(full) %in% c('PassengerId','Name','Survived')], method='rf') 
mice_output = complete(mice_mod)

#now that we have the new predicted values 
#we make sure its correct by comparing the distributions before and after 
par("mar")
par(mar=c(4,3,2,1))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04),xlab="age",ylab="density")
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04),xlab="age",ylab="density")
#the distribution remains largely unaltered , hence we can impute these values
full$Age = mice_output$Age
sum(is.na(full$Age))#number of missing values in age

#checking for any more missing values 
#excpet the 418 survived values in test data set 
sum(is.na(full)) - 418 # 0



#PEDICTION
#Data set is now complete , lets split back the data and build our model
train_data = full[1:891,]
test_data  = full[892:1309,]

#reduce columns Parch and SibSp into one column FamilySize
train_data$FamilySize = train_data$SibSp + train_data$Parch + 1
test_data$FamilySize  = test_data$SibSp + test_data$Parch + 1


#lets define a model accuracy function 
modelaccuracy = function(train_data, pre) {
  result_1 = train_data$Survived == pre
  sum(result_1) / length(pre)
}

#WE look at 2 diff types of predictive models and compare accuracy 

#Random Forest Model
#this model uses a forest of decision trees and combines them to come up with a better model
set.seed(984);
rf_model = randomForest(factor(Survived) ~ Pclass + Sex + Age  + 
                           Fare + Embarked + FamilySize,
                         data = train_data)

#Error rates 
par(mar=c(4,4,4,4))
plot(rf_model,ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    = importance(rf_model)
varImportance = data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance = varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

#checking model on the training data it self 
prediction   = predict(rf_model, train_data);
accuracy_RFM = modelaccuracy(train_data, prediction);
accuracy_RFM # 0.92031

#applying to test data 
prediction = predict(rf_model, test_data);

#filling in the predicted values and storing it in a file 
solution_RFM = data.frame(PassengerID = test_data$PassengerId, Survived = prediction);
write.csv(solution_RFM, file = 'RandomForestSolutions.csv', row.names = F)
#accuracy on kaggle 0.77990


#LOGISTIC REGRESSION Model

#applying to training data 
logitMod    = glm(Survived ~  Pclass+ Age + Sex + Embarked+ Fare+ FamilySize, data=train_data, family=binomial(link="logit"));
summary(logitMod)
prediction  = predict(logitMod, train_data, type="response")

#confussion matrix
con_matrix = table(train_data$Survived, prediction > 0.5)

accuracy_LRM = (con_matrix[1,1]+con_matrix[2,2])/length(prediction)
accuracy_LRM #0.80359

#ROC Curve
ROCRpred <- prediction(prediction, train_data$Survived)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
par("mar")
par(mar=c(4,4,4,4))
plot(ROCRperf, colorize = TRUE, ylab="True Positive Rate", xlab="False Positive Rate" )


#applying to test data
prediction = predict(logitMod, test_data, type="response")
solution_LRM = data.frame(PassengerID = test_data$PassengerId, Survived = round(prediction));
write.csv(solution_LRM, file = 'LogisticRegSolutions.csv', row.names = F)

#accuracy on kaggle 0.77033

