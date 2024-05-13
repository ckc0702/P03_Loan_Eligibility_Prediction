``
###----------------------------------------------Preparing Project----------------------------------------------
``
# https://www.kaggle.com/datasets/yasserh/loan-default-dataset

# rm(list = ls())


# Install Packages
install.packages(c("dplyr", "DataExplorer", "tidyverse", "ggplot2", "e1071", 
                   "caret", "quanteda", "irlba", "randomForest", "mice", 
                   "caTools", "splitstackshape", "performanceEstimation", 
                   "smotefamily", "lsr", "rpart", "rpart.plot", "ROCR", "mlr"))

install.packages("mlr")


# Load Packages - need to run in sequence (some overlapping function in mlr and other packages)
library(dplyr)                    # Grammar Manipulation
library(DataExplorer)             # Easier to visualize data
library(tidyverse)                # Query Like Structure
library(ggplot2)                  # Plotting
library(quanteda)                 # For text analytics
library(mice)                     # Imputation
library(performanceEstimation)    # SMOTE package 1
library(smotefamily)              # SMOTE package 2 
library(caret)                    # Classification training & Confusion Matrix
library(lsr)                      # Cramer V - correlation analysis
library(caTools)                  # Stratified Sampling
library(e1071)                    # SVM
library(randomForest)             # Random Forest
library(rpart)                    # Decision Tree
library(rpart.plot)               # Decision Tree Plot
library(ROCR)                     # Plot ROC
library(pROC)                     # For plotting ROC
library(mlr)                      # For Parameter Tuning

# Set Working Directory
setwd('C:/Users/USER/Desktop/University/Sem 1/Machine Learning/Assignment/Assignment Workdir/Dataset For Loan Prediction')

data <- read.csv('Dataset2/Loan_Default.csv', stringsAsFactors = FALSE)


``
###----------------------------------------------Exploratory Data Analysis (EDA)----------------------------------------------
``
df <- data

# Viewing data
dim(df) 
head(df, n=10)
summary(data)

# Drop columns
df <- drop_columns(df, c("ID", "year", "Interest_rate_spread"))
ncol(df)

# Characteristic of data
character_columns <- sapply(df, is.character)
lapply(df[, character_columns], unique)

# Convert "" to NA
df[df == ""] <- NA

# Check missing values
sum(is.na(df)) 
plot_missing(df) 
colSums(is.na(df))

# Convert target variable to factor
df$Status <- factor(df$Status, levels=c(0,1), labels=c("Rejected", "Approved")) 

class(df$Status)
summary(df$Status)

# Checking continous variable correlation
# No correlation >= 0.8
corr_data <- df[, -which(names(df) == "Status")]
corr_data <- na.omit(corr_data)
plot_correlation(corr_data,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# Checking independent variable class distribution
#*Some categories are very distinct, which affect chisq test result
table(df$Neg_ammortization)
barplot(prop.table(table(df$Neg_ammortization)), main = "Class Distribution", 
        xlab = "Status", ylab = "Percentage", col = "blue")

# Independent variables for chisq and fisher test
corr_data_discrete <- df[character_columns]
corr_data_discrete <- na.omit(corr_data_discrete)
corr_data_discrete_cols <- colnames(corr_data_discrete)

# Creating a dataframe to store chisq result
chisq_result_df <- data.frame(
  matrix(ncol = length(corr_data_discrete_cols), nrow = length(corr_data_discrete_cols))
)

# Appending the row and colname to the dataframe
colnames(chisq_result_df) <- rownames(chisq_result_df) <- corr_data_discrete_cols

# Checking discrete variable correlation - chisq test
for (i in 1:(length(corr_data_discrete_cols)-1)) {
  for (j in (i+1):length(corr_data_discrete_cols)) {
    var1 <- corr_data_discrete_cols[i]
    var2 <- corr_data_discrete_cols[j]
    
    # Create contingency table
    contingency_table <- table(corr_data_discrete[[var1]], 
                               corr_data_discrete[[var2]])
    
    # Perform chi-sq test
    chisq_result <- chisq.test(contingency_table)
    
    # Display result
    # cat("Chi-square test between", var1, "and", var2, ":\n")
    # print(chisq_result)
    # cat("\n")
    
    # Store p-value into corresponding cell
    chisq_result_df[var1, var2] <- chisq_result$p.value
    chisq_result_df[var2, var1] <- chisq_result$p.value
  }
}

# Creating a dataframe to store fisher result
fisher_result_df <- data.frame(
  matrix(ncol = length(corr_data_discrete_cols), nrow = length(corr_data_discrete_cols))
)

# Appending the row and colname to the dataframe
colnames(fisher_result_df) <- rownames(fisher_result_df) <- corr_data_discrete_cols


# Checking discrete variable correlation - fisher exact test
# https://stats.stackexchange.com/questions/81483/warning-in-r-chi-squared-approximation-may-be-incorrect
# https://stats.stackexchange.com/questions/67457/df-missing-in-r-output-of-chi-square-test
for (i in 1:(length(corr_data_discrete_cols)-1)) {
  for (j in (i+1):length(corr_data_discrete_cols)) {
    var1 <- corr_data_discrete_cols[i]
    var2 <- corr_data_discrete_cols[j]
    
    # Create contingency table
    contingency_table <- table(corr_data_discrete[[var1]], 
                               corr_data_discrete[[var2]])
    
    # Perform fisher test
    fisher_result <- chisq.test(contingency_table, simulate.p.value = TRUE)
    
    # Store p-value into corresponding cell
    fisher_result_df[var1, var2] <- fisher_result$p.value
    fisher_result_df[var2, var1] <- fisher_result$p.value
    
  }
}


# Creating a dataframe to store cramer result
cramer_result_df <- data.frame(
  matrix(ncol = length(corr_data_discrete_cols), nrow = length(corr_data_discrete_cols))
)

# Appending the row and colname to the dataframe
colnames(cramer_result_df) <- rownames(cramer_result_df) <- corr_data_discrete_cols

# Checking discrete variable correlation - cramer v test
for (i in 1:(length(corr_data_discrete_cols)-1)) {
  for (j in (i+1):length(corr_data_discrete_cols)) {
    var1 <- corr_data_discrete_cols[i]
    var2 <- corr_data_discrete_cols[j]
    
    # Create contingency table
    contingency_table <- table(corr_data_discrete[[var1]], 
                               corr_data_discrete[[var2]])
    
    # Perform cramer v test
    cramer_result <- round(cramersV(contingency_table),2)
    
    # Store p-value into corresponding cell
    cramer_result_df[var1, var2] <- cramer_result
    cramer_result_df[var2, var1] <- cramer_result
    
  }
}
View(chisq_result_df)
View(fisher_result_df)
View(cramer_result_df)


# Viewing discrete variables correlations results (CramerV)
#* Based on the observations, 
#* business_or_commercial have a fully associated relationship with loan_type = 1 
#* construction_type & security_by & security_type have very strong relationship = 0.98
#* 3 Element will have to be removed
#* Here I chose boc, construc_type & secured_by to remove
View(subset(cramer_result_df, apply(cramer_result_df, 1, function(x) any(x >= 0.8))))

df_new <- drop_columns(df, c("business_or_commercial", "construction_type", 
                         "Secured_by"))

# Since there are 148670 observations, I will remove columns with NA rows < 5% 
plot_missing(df_new)
df_new <- df_new[complete.cases(df_new[c("term", "loan_purpose", 
                                         "submission_of_application", "age", 
                                         "approv_in_adv", "Neg_ammortization",
                                         "loan_limit")]) == TRUE, ] 


``
###----------------------------------------------Data Cleaning----------------------------------------------
``

# Abnormalities in Loan To Value (LTV) should be within 100%
#* Based on observations,
#* For all LTV above 500 they have the same property value = 8000
#* This is abnormal pattern, hence those rows will be removed
subset(df_new, df_new$LTV > 500, select = c('property_value', 'loan_amount',
                                            'LTV'))
dt = subset(df_new, df_new$LTV <= 500 | is.na(df_new$LTV))

# Checking significance of gender to target variable
#* Gender is highly significant to target variable, so even though there are
#* a lots of invalid gender it is best not to remove this column and leave it be
valid_gender = dt[dt$Gender == "Male" | dt$Gender == "Female", ]
gender_status_contingency_table <- table(valid_gender$Gender,
                                         valid_gender$Status)
chisq.test(gender_status_contingency_table)

# Checking outliers for income
#* SD < Mean, 0.05% outliers, and normally distributed
#* Hence, I decided to use mean imputation instead of median
ignore_income_na <- na.omit(dt$income)
mean(ignore_income_na) 
sd(ignore_income_na)   
normalized_income <- log(ignore_income_na) 
plot(density(normalized_income)) 
boxplot(normalized_income)
income_iqr <- IQR(ignore_income_na) # Q3 - Q1
lower_boundary <- quantile(ignore_income_na, 0.25) - 1.5 * income_iqr
upper_boundary <- quantile(ignore_income_na, 0.75) + 1.5 * income_iqr
print(sum(ignore_income_na < lower_boundary | ignore_income_na > upper_boundary
          )) / length(ignore_income_na)


# Mean imputation
imputed_data <- dt
imputed_data$income <- ifelse(is.na(imputed_data$income), 
                              mean(imputed_data$income, na.rm = TRUE), 
                              imputed_data$income)
sum(is.na(imputed_data$income))

# Mice imputation
mice_impute <- mice(imputed_data, m = 1, seed=666)
mice_imputed_data = complete(mice_impute)
sum(is.na(mice_imputed_data))
plot_missing(mice_imputed_data)

# Convert Character to Factor
clean_data <- mice_imputed_data
characters_columns <- sapply(clean_data, is.character)
clean_data[characters_columns] <- lapply(clean_data[characters_columns], as.factor)
summary(clean_data)

``
###----------------------------------------------Data Splitting----------------------------------------------
``
# Split data into train and test set using stratified sampling
set.seed(123)
split <- sample.split(clean_data$Status, SplitRatio = 0.7)
training_set <- subset(clean_data, split == TRUE)
test_set <- subset(clean_data, split == FALSE)

converted_test_set <- test_set # For usage of ROC plotting
converted_test_set$Status <- factor(converted_test_set$Status, levels=c("Rejected","Approved"), labels=c(0, 1))

# Checking distribution in train and test set
#* Highly imbalance
prop.table(table(training_set$Status))
prop.table(table(test_set$Status))
table(training_set$Status)
barplot(prop.table(table(training_set$Status)), main = "Class Distribution", 
        xlab = "Status", ylab = "Percentage", col = "red")

# SMOTE oversampling
# https://stackoverflow.com/questions/54625093/smote-in-r-reducing-sample-size-significantly
# https://stackoverflow.com/questions/62871492/create-balanced-dataset-11-using-smote-without-modifying-the-observations-of-th
smote_data = smote(
  form = Status ~.,
  data = training_set,
  k = 5,
  perc.over = 2,
  perc.under = 0
)

# Adding synthetic minority data into training set for class balancing
# 76077 rejected - 24678 approved = 51399
bal_training_set <- rbind(training_set, smote_data[1:51399,])
prop.table(table(bal_training_set$Status))
barplot(prop.table(table(bal_training_set$Status)), main = "Class Distribution", 
        xlab = "Status", ylab = "Percentage", col = "red")


``
###----------------------------------------------Models Building----------------------------------------------
``
# Logistics Regression Models
# https://stackoverflow.com/questions/67360883/how-to-fix-fitted-probabilities-numerically-0-or-1-occurred-warning-in-r
logistics_classifier = glm(Status ~.,
                           data = training_set,
                           family = binomial)
summary(logistics_classifier)

logistics_classifier_bal = glm(Status ~.,
                               data = bal_training_set,
                               family = binomial)
summary(logistics_classifier_bal)

# Decision Tree Models
decision_tree_classifier = rpart(Status ~., data = training_set)
prp(decision_tree_classifier)
prp (decision_tree_classifier, type = 5, extra = 106)
rpart.plot(decision_tree_classifier, extra = 106, nn = TRUE)
printcp(decision_tree_classifier)
summary(decision_tree_classifier)

decision_tree_classifier_bal = rpart(Status ~., data = bal_training_set)
prp(decision_tree_classifier_bal)
prp (decision_tree_classifier_bal, type = 5, extra = 106)
rpart.plot(decision_tree_classifier_bal, extra = 106, nn = TRUE)
printcp(decision_tree_classifier_bal)

# Random Forest Models
random_forest_classifier = randomForest(Status ~., data = training_set)
print(random_forest_classifier)
attributes(random_forest_classifier) # the variables used for building the trees
random_forest_classifier$ntree

random_forest_classifier_bal = randomForest(Status ~., data = bal_training_set)
print(random_forest_classifier_bal)
attributes(random_forest_classifier_bal)
random_forest_classifier_bal$ntree

``
###----------------------------------------------Logistics Regression Model Evaluation----------------------------------------------
``
#***1***
# Predicting the Training set results
logit_pred_prob_training <- predict(logistics_classifier, type = 'response', 
                                    training_set[ ,-30] )
logit_pred_class_training = ifelse(logit_pred_prob_training > 0.5, 1, 0)
logit_pred_class_training

# Reorder the variable
logit_pred_class_training <- factor(logit_pred_class_training, levels=c(0,1),
                                    labels=c("Rejected", "Approved"))

logit_cm_training = table(training_set$Status, logit_pred_class_training)
logit_cm_training

# Confusion Matrix for training set
based_logit_unbalance_train_cm = confusionMatrix(logit_cm_training)

#***2***
# Predicting the Test set results
logit_pred_prob_test <- predict(logistics_classifier, type = 'response',
                                test_set[ ,-30] )
logit_pred_class_test = ifelse(logit_pred_prob_test > 0.5, 1, 0)
logit_pred_class_test

# Reorder the variable
logit_pred_class_test <- factor(logit_pred_class_test, levels=c(0,1),
                                labels=c("Rejected", "Approved"))

logit_cm_test = table(test_set$Status, logit_pred_class_test)
logit_cm_test

# Confusion Matrix for test set
based_logit_unbalance_test_cm = confusionMatrix(logit_cm_test);based_logit_unbalance_test_cm

#***3***
# Predicting the Test set results using Model trained with balanced data
logit_bal_pred_prob_test <- predict(logistics_classifier_bal, type = 'response',
                                test_set[ ,-30] )
logit_bal_pred_class_test = ifelse(logit_bal_pred_prob_test > 0.5, 1, 0)
logit_bal_pred_class_test

# Reorder the variable
logit_bal_pred_class_test <- factor(logit_bal_pred_class_test, levels=c(0,1),
                                labels=c("Rejected", "Approved"))

logit_bal_cm_test = table(test_set$Status, logit_bal_pred_class_test)
logit_bal_cm_test

# Confusion Matrix for test set
based_logit_cm = confusionMatrix(logit_bal_cm_test);based_logit_cm

#***Accuracy For Logistic Regression***
logit_base_accuracy_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Accuracy = c(round(based_logit_unbalance_train_cm$overall['Accuracy'], 2),
               round(based_logit_unbalance_test_cm$overall['Accuracy'], 2),
               round(based_logit_cm$overall['Accuracy'], 2))
)
logit_base_accuracy_df$Types <- factor(logit_base_accuracy_df$Types, levels =
                                         c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(logit_base_accuracy_df, aes(x = Types, y = Accuracy, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")

#***Specificity for Logistic Regression***
logit_base_specificity_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Specificity = c(round(based_logit_unbalance_train_cm$byClass['Specificity'], 2),
               round(based_logit_unbalance_test_cm$byClass['Specificity'], 2),
               round(based_logit_cm$byClass['Specificity'], 2))
)
logit_base_specificity_df$Types <- factor(logit_base_specificity_df$Types, levels =
                                         c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(logit_base_specificity_df, aes(x = Types, y = Specificity, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")


#***ROC***
logit_pred = prediction(logit_bal_pred_prob_test, converted_test_set$Status)
logit_perf = ROCR::performance(logit_pred, "tpr", "fpr")
plot(logit_perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

#***AUC***
logit_auc <- as.numeric(ROCR::performance(logit_pred, "auc")@y.values)
logit_auc <-  round(logit_auc, 2)
logit_auc # 0.84


``
###----------------------------------------------Decision Tree Model Evaluation----------------------------------------------
``
#***1***
# Predicting the Training set results
dt_pred_prob_training <- predict(decision_tree_classifier, training_set)

# Predict class instead of prob, threshold default 0.5
dt_pred_class_training <- predict(decision_tree_classifier, training_set,
                                  type = "class")

dt_cm_training = table(training_set$Status, dt_pred_class_training)

# Confusion Matrix for training set
base_dt_unbalance_train_cm = confusionMatrix(dt_cm_training)

#***2***
# Predicting the Test set results
dt_pred_prob_test <- predict(decision_tree_classifier, test_set)

# Predict class instead of prob, threshold default 0.5
dt_pred_class_test <- predict(decision_tree_classifier, test_set,
                                  type = "class")

dt_cm_test = table(test_set$Status, dt_pred_class_test)

# Confusion Matrix for testing set
base_dt_unbalance_test_cm = confusionMatrix(dt_cm_test)


#***3***
# Predicting the Test set results using Model trained with balanced data
dt_bal_pred_prob_test <- predict(decision_tree_classifier_bal, test_set)

# Predict class instead of prob, threshold default 0.5
dt_bal_pred_class_test <- predict(decision_tree_classifier_bal, test_set,
                              type = "class")

dt_bal_cm_test = table(test_set$Status, dt_bal_pred_class_test)

# Confusion Matrix for testing set
base_dt_cm = confusionMatrix(dt_bal_cm_test)

#***Accuracy For Decision Tree***
dt_base_accuracy_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Accuracy = c(round(base_dt_unbalance_train_cm$overall['Accuracy'], 2),
               round(base_dt_unbalance_test_cm$overall['Accuracy'], 2),
               round(base_dt_cm$overall['Accuracy'], 2))
)
dt_base_accuracy_df$Types <- factor(dt_base_accuracy_df$Types, levels =
                                         c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(dt_base_accuracy_df, aes(x = Types, y = Accuracy, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")

#***Specificity For Decision Tree***
dt_base_specificity_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Specificity = c(round(base_dt_unbalance_train_cm$byClass['Specificity'], 2),
                  round(base_dt_unbalance_test_cm$byClass['Specificity'], 2),
                  round(base_dt_cm$byClass['Specificity'], 2))
)
dt_base_specificity_df$Types <- factor(dt_base_specificity_df$Types, levels =
                                            c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(dt_base_specificity_df, aes(x = Types, y = Specificity, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")


#***ROC***
dt_pred = prediction(dt_bal_pred_prob_test[,2], converted_test_set$Status)
dt_perf = ROCR::performance(dt_pred, "tpr", "fpr")
plot(dt_perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

#***AUC***
dt_auc <- as.numeric(ROCR::performance(dt_pred, "auc")@y.values)
dt_auc <-  round(dt_auc, 2)
dt_auc # 0.81

``
###----------------------------------------------Random Forest Model Evaluation----------------------------------------------
``
#***1***
# Predicting the Training set results
rf_pred_class_training <- predict(random_forest_classifier, training_set)

rf_cm_train = table(training_set$Status, rf_pred_class_training)

# Confusion Matrix for training set
base_rf_unbalance_train_cm = confusionMatrix(rf_cm_train)

#***2***
# Predicting the Test set results
rf_pred_class_test <- predict(random_forest_classifier, test_set)

rf_cm_test = table(test_set$Status, rf_pred_class_test)

# Confusion Matrix for training set
base_rf_unbalance_test_cm = confusionMatrix(rf_cm_test)

#***3***
# Predicting the Test set results using Model trained with balanced data
rf_bal_pred_class_test <- predict(random_forest_classifier_bal, test_set)

rf_bal_cm_test = table(test_set$Status, rf_bal_pred_class_test)

# Confusion Matrix for testing set
base_rf_cm = confusionMatrix(rf_bal_cm_test)

#***Accuracy For Random Forest***
rf_base_accuracy_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Accuracy = c(round(base_rf_unbalance_train_cm$overall['Accuracy'], 2),
               round(base_rf_unbalance_test_cm$overall['Accuracy'], 2),
               round(base_rf_cm$overall['Accuracy'], 2))
)
rf_base_accuracy_df$Types <- factor(rf_base_accuracy_df$Types, levels =
                                      c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(rf_base_accuracy_df, aes(x = Types, y = Accuracy, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")

#***Specificity For Random Forest***
rf_base_specificity_df <- data.frame(
  Types = c("Training Set", "Test Set", "Test Set (bal)"),
  Specificity = c(round(base_rf_unbalance_train_cm$byClass['Specificity'], 2),
                  round(base_rf_unbalance_test_cm$byClass['Specificity'], 2),
                  round(base_rf_cm$byClass['Specificity'], 2))
)
rf_base_specificity_df$Types <- factor(rf_base_specificity_df$Types, levels =
                                         c("Training Set", "Test Set", "Test Set (bal)")) # sorting
ggplot(rf_base_specificity_df, aes(x = Types, y = Specificity, fill = Types)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")

``
###----------------------------------------------Base Models Comparison----------------------------------------------
``
#***Accuracy Comparison***
base_models_accuracy_df <- data.frame(
  Models = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(round(based_logit_cm$overall['Accuracy'], 4),
               round(base_dt_cm$overall['Accuracy'], 4),
               round(base_rf_cm$overall['Accuracy'], 4))
)
# Sort by accuracy
base_models_accuracy_df <- 
  base_models_accuracy_df[order(base_models_accuracy_df$Accuracy),]
# Plotting
ggplot(base_models_accuracy_df, aes(x = Models, y = Accuracy, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")


#***Specificity Comparison***
base_models_specificity_df <- data.frame(
  Models = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Specificity = c(round(based_logit_cm$byClass['Specificity'], 4),
               round(base_dt_cm$byClass['Specificity'], 4),
               round(base_rf_cm$byClass['Specificity'], 4))
)
# Sort by specificity
base_models_specificity_df <- 
  base_models_specificity_df[order(base_models_specificity_df$Specificity),]
# Plotting
ggplot(base_models_specificity_df, aes(x = reorder(Models, Specificity),
                                       y = Specificity, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")


``
###----------------------------------------------Cross Validation for Logistics Regression (GLM)----------------------------------------------
``
# (repeated 5-fold cross-validation)
control <- trainControl(method = "repeatedcv",
                    search = "grid",
                    number = 5,
                    repeats = 5, 
                    verboseIter = T)

caret.logit_cv <- train(Status ~.,
                        bal_training_set,
                        method = 'glm',
                        trControl = control,
                        metric = 'Accuracy')
caret.logit_cv # 0.7715

``
###----------------------------------------------Model Tuning & Cross Validation----------------------------------------------
``
#* https://towardsdatascience.com/decision-tree-hyperparameter-tuning-in-r-using-mlr-3248bfd2d88c
#* https://stackoverflow.com/questions/47822694/logistic-regression-tuning-parameter-grid-in-r-caret-package

# View tunable parameter for models
#* Logistic regression have no tune parameter
getParamSet("classif.logreg")
getParamSet("classif.rpart")
getParamSet("classif.randomForest")

# Define tune task
mlr.training.task <- makeClassifTask(
  data = bal_training_set,
  target = "Status"
)

# Define tuning control grid
mlr.tune.control_grid <- makeTuneControlGrid()

# Define cross validation (repeated 5-fold cross-validation)
mlr.resample <- makeResampleDesc("CV", iters = 5) 

# Define tuning measure
mlr.measure = acc

# Define param grid 
mlr.dt.params_grid <- makeParamSet(
  makeDiscreteParam("maxdepth", values = 25:30),
  makeDiscreteParam("minsplit", values = 15:20),
  makeNumericParam("cp", lower = 0.001, upper = 0.01)
)

#* ntree - 500, mtry - regression = p/3 & class = sqrt(p), nodesize - def 1, related to tree depth, higher number, lower depth
mlr.rf.params_grid <- makeParamSet(
  makeDiscreteParam("ntree", values = c(500, 510, 520, 540, 560, 600, 620, 640)),
  makeDiscreteParam("nodesize", values = 1)
) 

mlr.rf.params_grid_2 <- makeParamSet(
  makeDiscreteParam("ntree", values = c(500, 504, 520)),
  makeDiscreteParam("mtry", values = 1:4),
  makeDiscreteParam("nodesize", values = 1:4)
) 


set.seed(515)
# Testing repeated 5 folds using MLR for logistic regression
mlr.logit_cv <- resample(learner = 'classif.logreg',
                         task = mlr.training.task,
                         measure = mlr.measure,
                         resampling = mlr.resample,
                         show.info = TRUE) # 0.7715

# Hyperparameter tuning
mlr.dt_tuned_hyperparam <- tuneParams(learner='classif.rpart',
                                  task = mlr.training.task,
                                  resampling = mlr.resample,
                                  measure = mlr.measure,
                                  par.set = mlr.dt.params_grid,
                                  control = mlr.tune.control_grid,
                                  show.info = TRUE) 
mlr.dt_tuned_hyperparam 

mlr.rf_tuned_hyperparam <- tuneParams(learner='classif.randomForest',
                                      task = mlr.training.task,
                                      resampling = mlr.resample,
                                      measure = mlr.measure,
                                      par.set = mlr.rf.params_grid_2,
                                      control = mlr.tune.control_grid,
                                      show.info = TRUE)
mlr.rf_tuned_hyperparam 


mlr.rf_tuned_hyperparam_retuned <- tuneParams(learner='classif.randomForest',
                                      task = mlr.training.task,
                                      resampling = mlr.resample,
                                      measure = mlr.measure,
                                      par.set = mlr.rf.params_grid,
                                      control = mlr.tune.control_grid,
                                      show.info = TRUE)
mlr.rf_tuned_hyperparam_retuned 

# Building classifier with most optimal param
mlr.dt_best_param <- setHyperPars(
  makeLearner("classif.rpart"),
  par.vals = mlr.dt_tuned_hyperparam$x
)
dt_tuned_classfier <- mlr::train(mlr.dt_best_param, mlr.training.task)


mlr.rf_param_tuned <- setHyperPars(
  makeLearner("classif.randomForest"),
  par.vals = mlr.rf_tuned_hyperparam$x
)
rf_tuned_classifier <- mlr::train(mlr.rf_param_tuned, mlr.training.task)

mlr.rf_best_param <- setHyperPars(
  makeLearner("classif.randomForest"),
  par.vals = mlr.rf_tuned_hyperparam_retune$x
)
rf_best_tuned_classifier <- mlr::train(mlr.rf_best_param, mlr.training.task)

``
###----------------------------------------------Models Validation (Optimized)----------------------------------------------
``
# Create predict task
mlr.test.task <- makeClassifTask(
  data = test_set,
  target = "Status"
)

#***Logistics Regression***
logit_pred_class <- predict(caret.logit_cv, test_set[ ,-30])

# Creating contingency table
logit_cm = table(test_set$Status, logit_pred_class)

# Confusion Matrix for training set
tuned_logit_cm = confusionMatrix(logit_cm)


#***Decision Tree***
dt_pred_class <- predict(dt_tuned_classfier, mlr.test.task)$data

# Create contingency table for actual and predicted
dt_cm = table(dt_pred_class$truth, dt_pred_class$response)

# Create confusion matrix
tuned_dt_cm = confusionMatrix(dt_cm)


#***Random Forest***
rf_pred_class <- predict(rf_best_tuned_classifier, mlr.test.task)$data

# Create contingency table for actual and predicted
rf_cm = table(rf_pred_class$truth, rf_pred_class$response)

# Create confusion matrix
tuned_rf_cm = confusionMatrix(rf_cm); 

``
###----------------------------------------------Tuned vs Base Models Comparison----------------------------------------------
``
#***Accuracy Comparison***
# Logistics Regression Tuned vs Base Model
logit_models_accuracy_df <- data.frame(
  Models = c("Tuned", "Base"),
  Accuracy = c(round(tuned_logit_cm$overall['Accuracy'], 4),
               round(based_logit_cm$overall['Accuracy'], 4))
)
ggplot(logit_models_accuracy_df, aes(x = Models, y = Accuracy, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")

# Decision Tree Tuned vs Base Model
dt_models_accuracy_df <- data.frame(
  Models = c("Tuned", "Base"),
  Accuracy = c(round(tuned_dt_cm$overall['Accuracy'], 4),
               round(base_dt_cm$overall['Accuracy'], 4))
)
ggplot(dt_models_accuracy_df, aes(x = Models, y = Accuracy, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")


# Random Forest Tuned vs Base Model
rf_models_accuracy_df <- data.frame(
  Models = c("Tuned", "Base"),
  Accuracy = c(round(tuned_rf_cm$overall['Accuracy'], 4),
               round(base_rf_cm$overall['Accuracy'], 4))
)
ggplot(rf_models_accuracy_df, aes(x = Models, y = Accuracy, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")


#***Specificity Comparison***
# Logistics Regression Tuned vs Base Model
logit_models_specificity_df <- data.frame(
  Models = c("Tuned", "Base"),
  Specificity = c(round(tuned_logit_cm$byClass['Specificity'], 4),
               round(based_logit_cm$byClass['Specificity'], 4))
)
ggplot(logit_models_specificity_df, aes(x = Models, y = Specificity, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")

# Decision Tree Tuned vs Base Model
dt_models_specificity_df <- data.frame(
  Models = c("Tuned", "Base"),
  Specificity = c(round(tuned_dt_cm$byClass['Specificity'], 4),
               round(base_dt_cm$byClass['Specificity'], 4))
)
ggplot(dt_models_specificity_df, aes(x = Models, y = Specificity, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")


# Random Forest Tuned vs Base Model
rf_models_specificity_df <- data.frame(
  Models = c("Tuned", "Base"),
  Specificity = c(round(tuned_rf_cm$byClass['Specificity'], 4),
               round(base_rf_cm$byClass['Specificity'], 4))
)
ggplot(rf_models_specificity_df, aes(x = Models, y = Specificity, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")


``
###----------------------------------------------Comparison Between Tuned Models----------------------------------------------
``
#***Accuracy Comparison***
tuned_models_accuracy_df <- data.frame(
  Models = c("Logistics Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(round(tuned_logit_cm$overall['Accuracy'], 2),
               round(tuned_dt_cm$overall['Accuracy'], 2),
               round(tuned_rf_cm$overall['Accuracy'], 2))
)
# Sort by accuracy
tuned_models_accuracy_df <- tuned_models_accuracy_df[order(tuned_models_accuracy_df$Accuracy),]
# Plotting
ggplot(tuned_models_accuracy_df, aes(x = reorder(Models, Accuracy),
                                     y = Accuracy, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Accuracy), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Accuracy")

#***Specificity Comparison***
tuned_models_specificity_df <- data.frame(
  Models = c("Logistics Regression", "Decision Tree", "Random Forest"),
  Specificity = c(round(tuned_logit_cm$byClass['Specificity'], 2),
               round(tuned_dt_cm$byClass['Specificity'], 2),
               round(tuned_rf_cm$byClass['Specificity'], 2))
)
# Sort by accuracy
tuned_models_specificity_df <- tuned_models_specificity_df[order(tuned_models_specificity_df$Specificity),]
# Plotting
ggplot(tuned_models_specificity_df, aes(x = reorder(Models, Specificity),
                                     y = Specificity, fill = Models)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Specificity), vjust = -0.5, color = "black") +
  labs(x = NULL, y = "Specificity")
