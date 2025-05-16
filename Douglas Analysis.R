# Douglas Chew Xi Zhi TP075339

library(ggplot2)
library(vcd)
library(dplyr)
library(caret)
library(VIM)
library(caTools)
library(randomForest)
library(Matrix)
library(xgboost)
library(pROC)
library(car)
library(DescTools)
library(psych)
library(shapviz)

## Data Import ##
data = read.csv("cleaned_bank_set.csv")

data$X <- NULL
data$class <- factor(data$class, levels = c("bad", "good"))

# Objective: To investigate the relationship between duration and purpose with credit class
##################################################################################################################
### Question 1: What is the pattern between duration, purpose and credit class? ###
## Boxplot without aggregation of purpose ##
ggplot(data, aes(x = interaction(purpose, class), y = duration, fill = class)) +
  geom_boxplot() +
  scale_fill_discrete(labels = c("Good", "Bad")) +
  labs(title = "Boxplot of Duration by Purpose and Class", x = "Purpose and class", y = "Duration") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

## Boxplot with aggregation of purpose ##
ggplot(data, aes(x = class, y = duration, fill = class)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "text", aes(label = paste("Mean :", round(..y.., 0))), vjust = -0.2) +
  labs(title = "Boxplot of Duration by Class", x = "Class", y = "Duration")

## Violin plot ##
ggplot(data, aes(x = "" ,y = duration, fill = class)) +
  geom_violin(alpha = 0.4, position = "identity") +
  labs(title = "Distribution of Duration by Class Across Different Purposes", x = "", y = "Duration") +
  facet_wrap(~purpose) +
  theme(strip.text = element_text(size = 13))

## Heatmap ##
ggplot(data, aes(x = class, y = purpose, fill = duration)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "cyan", high = "red") +
  labs(title = "Heatmap of Aggregated Continuous IV by Categorical IV and DV",
       x = "DV", y = "Categorical IV") +
  theme_minimal()

## Scatter plot with jitter ##
ggplot(data, aes(x = duration, y = purpose, color = class, shape = class)) +
  geom_jitter(width = 0.2, height = 0.05) +
  labs(title = "Scatter Plot of Duration by Credit Class", x = "Duration", y = "Credit Class (Binary)") +
  theme_minimal()

## Density plot ##
ggplot(data, aes(x = duration, fill = class)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ purpose) +
  labs(title = "Density Plot of Duration by Credit Class and Purpose", x = "Duration", y = "Density")


##################################################################################################################
### Question 2: Is there any association between purpose and duration with class? ###
## Hypothesis testing for purpose and class ##
# Chi-Square Test of Independence #
# Create a contingency table
contingency_table <- table(data$purpose, data$class)

# Perform Chi-square test
chisq.test(contingency_table)

## Test strength of association for purpose and class ##
# Cramer's V #
CramerV(contingency_table)

## Hypothesis testing for duration and class ##
# Logistic regression #
# Train logistic regression
lr_model <- train(class ~ duration, data = data, method = "glm", family = binomial)

# Get p-value
summary(lr_model)

## Correlation for duration and class ##
# Convert credit class into binary
binary_class <- ifelse(data$class == "bad", 1, 0)

# Point-biserial correlation
biserial(data$duration, binary_class)


##################################################################################################################
### Question 3: Which models perform well when using duration and purpose as predictors for determining credit classification?  ###
# Ensure reproducibility
set.seed(1111)

## Train test split ##
# Split into two dataset which 80% for training while 20% for testing
split = sample.split(data$class, SplitRatio = 0.8)
training_set = subset(data, split == T)
test_set = subset(data, split == F)


##################################################################################################################
## K-Fold Cross Validation ##
# Setup K-Fold Cross Validation as 10 fold
train_control <- trainControl(method = "cv", number = 10)


##################################################################################################################
## Logistic Regression ##
# Train logistic regression
lr_model <- train(class ~ duration + purpose, data = training_set, method = "glm", family = binomial, trControl = train_control)

# Compute Metrics of Logistic Regression #
# For training set #
# Predict label
lr_train_predict <- predict(lr_model, newdata = training_set[,-21])

# Creaete confusion matrix
lr_train_cm <- confusionMatrix(table(lr_train_predict, training_set$class))
lr_train_cm

# For test set #
# Predict label
lr_test_predict <- predict(lr_model, newdata = test_set[,-21], type = "prob")

# Create ROC object
lr_roc <- roc(test_set$class, lr_test_predict[,1])

# Label the prediction
lr_test_predict <- ifelse(lr_test_predict[,1] >= 0.5, "bad", "good")

# Confusion matrix
lr_test_cm <- confusionMatrix(table(lr_test_predict, test_set$class))
lr_test_cm

# Summary of Logistic Regression #
# Metrics of Logistic Regression
lr_evaluation <- data.frame(
  Train_Accuracy = lr_train_cm$overall["Accuracy"],
  Test_Accuracy = lr_test_cm$overall["Accuracy"],
  Precision = lr_test_cm$byClass['Precision'],
  Recall = lr_test_cm$byClass['Recall'],
  F1_Score = lr_test_cm$byClass['F1'],
  AUROC = auc(lr_roc)
)
rownames(lr_evaluation) <- c("Logistic Regression")
lr_evaluation

# Coefficient summary
summary(lr_model)


##################################################################################################################
## Random forest ##
# Train random forest model
rf_model <- train(class ~ purpose + duration, data = training_set, method = "rf", trControl = train_control, ntree = 500)

# Compute Metrics of Random Forest #
# For training set #
# Predict label
rf_train_predict <- predict(rf_model, newdata = training_set[,-21])

# Create confusion matrix
rf_train_cm <- confusionMatrix(table(rf_train_predict, training_set$class))
rf_train_cm

# For test set #
# Predict label
rf_test_predict <- predict(rf_model, newdata = test_set[,-21], type = "prob")

# Create ROC object
rf_roc <- roc(test_set$class, rf_test_predict[,1])

# Label prediction
rf_test_predict <- ifelse(rf_test_predict[,1] >= 0.5, "bad", "good")

# Create confusion matrix
rf_test_cm <- confusionMatrix(table(rf_test_predict, test_set$class))
rf_test_cm

# Summary of Random Forest #
# Metrics of Random Forest
rf_evaluation <- data.frame(
  Train_Accuracy = rf_train_cm$overall["Accuracy"],
  Test_Accuracy = rf_test_cm$overall["Accuracy"],
  Precision = rf_test_cm$byClass['Precision'],
  Recall = rf_test_cm$byClass['Recall'],
  F1_Score = rf_test_cm$byClass['F1'],
  AUROC = auc(rf_roc)
)
rownames(rf_evaluation) <- c("Random Forest")
rf_evaluation


##################################################################################################################
## XGBoost ##
# Setup feature and label #
# Features
features <- c("duration", "purpose")

# Separate feature and label
X_train <- model.matrix(~ . - 1, data = training_set[,features])
y_train <- ifelse(as.numeric(training_set$class) == 2, 0, 1)
X_test <- model.matrix(~ . - 1, data = test_set[,features])
y_test <- ifelse(as.numeric(test_set$class) == 2, 0, 1)

# Create DMatrix
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Ensure reproducibility
set.seed(1111)

## XGBoost ##
# Cross validation #
cv_results <- xgb.cv(data = dtrain, nrounds = 100, nfold = 10, verbose = FALSE, early_stopping_rounds = 10)

# Train XGBoost model
xgb_model <- xgb.train(data = dtrain, nrounds = cv_results$best_iteration, lambda = 1)

# Compute Metrics of XGBoost #
# For training set #
# Predict label
xgb_train_predict <- predict(xgb_model, newdata = X_train)

# Label prediction
xgb_train_predict <- ifelse(xgb_train_predict >= 0.5, 1, 0)

# Create confusion matrix
xgb_train_cm <- confusionMatrix(table(xgb_train_predict, y_train), positive = "1")
xgb_train_cm

# For test set #
# Predict label
xgb_test_predict <- predict(xgb_model, newdata = X_test)

# Create ROC object
xgb_roc <- roc(y_test, xgb_test_predict)

# Label prediction
xgb_test_predict <- ifelse(xgb_test_predict >= 0.5, 1, 0)

# Create confusion matrix
xgb_test_cm <- confusionMatrix(table(xgb_test_predict, y_test), positive = "1")
xgb_test_cm

# Summary of XGBoost #
# Metrics of XGBoost 
xgb_evaluation <- data.frame(
  Train_Accuracy = xgb_train_cm$overall["Accuracy"],
  Test_Accuracy = xgb_test_cm$overall["Accuracy"],
  Precision = xgb_test_cm$byClass['Precision'],
  Recall = xgb_test_cm$byClass['Recall'],
  F1_Score = xgb_test_cm$byClass['F1'],
  AUROC = auc(xgb_roc)
)
rownames(xgb_evaluation) <- c("XGBoost")
xgb_evaluation

##################################################################################################################
## ROC Curve ##
# Plot logistic regression ROC curve
plot(lr_roc, col = "green" ,main = "ROC Curves")

# Plot random forest ROC curve
lines(rf_roc, col = "blue")

# Plot XGBoost ROC curve
lines(xgb_roc, col = "red")

# Add legend
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "XGBoost"), col = c("green", "blue", "red"), lwd = 2)


##################################################################################################################
## Model Evaluation ##
model_evaluation <- rbind(lr_evaluation, rf_evaluation)
model_evaluation <- rbind(model_evaluation, xgb_evaluation)
model_evaluation


##################################################################################################################
### Question 4: Does credit amount improve the performance of XGBoost? ###
## XGBoost New ##
# Setup feature and label #
# Features
features_2 <- c("duration", "purpose", "credit_amount")

# Separate feature and label
X_train_2 <- model.matrix(~ . - 1, data = training_set[,features_2])
y_train_2 <- ifelse(as.numeric(training_set$class) == 2, 0, 1)
X_test_2 <- model.matrix(~ . - 1, data = test_set[,features_2])
y_test_2 <- ifelse(as.numeric(test_set$class) == 2, 0, 1)

# Create DMatrix
dtrain_2 <- xgb.DMatrix(data = X_train_2, label = y_train_2)
dtest_2 <- xgb.DMatrix(data = X_test_2, label = y_test_2)

# Ensure reproducibility
set.seed(1111)

## XGBoost ##
# Cross validation #
cv_results <- xgb.cv(data = dtrain_2, nrounds = 100, nfold = 10, verbose = FALSE, early_stopping_rounds = 10)

# Train XGBoost model
xgb_model_2 <- xgb.train(data = dtrain_2, nrounds = cv_results$best_iteration, lambda = 1)

# Compute Metrics of XGBoost #
# For training set #
# Predict label
xgb_train_predict_2 <- predict(xgb_model_2, newdata = X_train_2)

# Label prediction
xgb_train_predict_2 <- ifelse(xgb_train_predict_2 >= 0.5, 1, 0)

# Create confusion matrix
xgb_train_cm_2 <- confusionMatrix(table(xgb_train_predict_2, y_train_2), positive = "1")
xgb_train_cm_2

# For test set #
# Predict label
xgb_test_predict_2 <- predict(xgb_model_2, newdata = X_test_2)

# Create ROC object
xgb_roc_2 <- roc(y_test_2, xgb_test_predict_2)

# Label prediction
xgb_test_predict_2 <- ifelse(xgb_test_predict_2 >= 0.5, 1, 0)

# Create confusion matrix
xgb_test_cm_2 <- confusionMatrix(table(xgb_test_predict_2, y_test_2), positive = "1")
xgb_test_cm_2

# Summary of XGBoost #
# Metrics of XGBoost 
xgb_evaluation_2 <- data.frame(
  Train_Accuracy = xgb_train_cm_2$overall["Accuracy"],
  Test_Accuracy = xgb_test_cm_2$overall["Accuracy"],
  Precision = xgb_test_cm_2$byClass['Precision'],
  Recall = xgb_test_cm_2$byClass['Recall'],
  F1_Score = xgb_test_cm_2$byClass['F1'],
  AUROC = auc(xgb_roc_2)
)
rownames(xgb_evaluation_2) <- c("XGBoost")
xgb_evaluation_2


##################################################################################################################
## ROC Curve for comparing XGBoost model after credit amount as predictor ##
# Plot logistic regression ROC curve
plot(xgb_roc, col = "red" ,main = "ROC Curves")

# Plot random forest ROC curve
lines(xgb_roc_2, col = "blue")

# Add legend
legend("bottomright", legend = c("Old XGBoost", "New XGBoost"), col = c("red", "blue"), lwd = 2)


##################################################################################################################
## Model Evaluation ##
model_evaluation <- rbind(xgb_evaluation, xgb_evaluation_2)
model_evaluation
model.matrix(~ purpose = ifelse(purpose == "business", 1, 0), data = training_set)
model.matrix(~ . - 1, data = training_set[,features_2])

##################################################################################################################
## Optimize Technique for duration ##
optimize_duration <- function(purpose, credit_amount, model, min_duration = 6, max_duration = 60) {
  # Generate a range of durations
  durations <- seq(min_duration, max_duration, by = 1)
  
  # Create a data frame for predictions
  test_data <- data.frame(
    duration = durations,
    credit_amount = rep(credit_amount, length(durations))
  )
  
  # One hot encoding for purpose
  for (p in unique(training_set$purpose)) {
    if (p != purpose) {
      test_data <- cbind(test_data, rep(0, length(durations)))
    } else {
      test_data <- cbind(test_data, rep(1, length(durations)))
    }
    
    names(test_data)[length(test_data)] <- paste("purpose", p, sep = "")
  }
  
  # Create matrix and realign feature names
  test_matrix = as.matrix(test_data)
  test_matrix <- test_matrix[, xgb_model_2$feature_names, drop = FALSE]
  
  # Predict probabilities for each duration
  test_data$default_prob <- predict(model, newdata = test_matrix, type = "response")
  
  # Find the duration with the lowest probability
  optimal_duration <- test_data$duration[which.min(test_data$default_prob)]
  min_probability <- min(test_data$default_prob)
  
  # Return recommendation
  if (min_probability >= 0.5) {
    return (paste("Reject loans because minimum probability of risk is", round(min_probability, 2)))
  } else {
    return(list(optimal_duration = optimal_duration, min_probability = min_probability))
  }
}

# Compare different purpose
optimize_duration("business", 6000, xgb_model_2)
optimize_duration("used car", 6000, xgb_model_2)

optimize_duration("education", 4000, xgb_model_2)
optimize_duration("education", 3000, xgb_model_2)

optimize_duration("business", 300000, xgb_model_2)


##################################################################################################################
## Grid search for hyperparameter tuning of XGBoost ##
# Define parameter grid
grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1),
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = c(1, 5),
  subsample = c(0.8, 1)
)

# Initialize variables to store grid search result
best_auc <- 0
best_params <- list()

# Grid search for hyperparameter tuning
for (i in 1:nrow(grid)) {
  params <- as.list(grid[i, ])
  params$objective <- "binary:logistic"
  params$eval_metric <- "auc"
  
  # Cross-validation
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    verbose = FALSE,
    early_stopping_rounds = 10
  )
  
  # Update best parameters if current AUC is better
  mean_auc <- max(cv_results$evaluation_log$test_auc_mean)
  if (mean_auc > best_auc) {
    best_auc <- mean_auc
    best_params <- params
  }
}

# Tune XGBoost
params <- list(
  objective = best_params$objective,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  lambda = 1
)


##################################################################################################################
## Plot probability
xgb_test_predict <- predict(xgb_model, newdata = dtest)

result <- test_set %>%
  mutate(bad_prob = xgb_test_predict) %>%
  mutate(class = ifelse(class == "bad", 1, 0)) %>%
  filter(ifelse(bad_prob >= 0.5, 1, 0) == class) %>%
  filter(purpose == "radio/tv")

ggplot(result, aes(x = duration, y = bad_prob, color = purpose)) +
  geom_point()


##################################################################################################################
## SHAP value for XGBoost ##
# SHAP
# Compute SHAP value with XGBoost model
shap <- shapviz(xgb_model_2, X_train_2)

## Combined SHAP values of purpose ##
# Select all columns for purpose
purpose = list(purpose = colnames(shap)[grepl('purpose', colnames(shap))])

# Collapse SHAP values
combined_shap <- collapse_shap(shap$S, collapse = purpose)
combined_shap <- shapviz(combined_shap, training_set[,-21])

# Plot SHAP importance
sv_importance(combined_shap)
sv_dependence(combined_shap, "duration", "purpose")

# Plot Depedence
sv_dependence(shap, "duration", "purposebusiness")
sv_dependence(shap, "duration", "purposedomestic appliance")
sv_dependence(shap, "duration", "purposeeducation")
sv_dependence(shap, "duration", "purposefurniture/equipment")
sv_dependence(shap, "duration", "purposenew car")
sv_dependence(shap, "duration", "purposeother")
sv_dependence(shap, "duration", "purposeradio/tv")
sv_dependence(shap, "duration", "purposerepairs")
sv_dependence(shap, "duration", "purposeretraining")
sv_dependence(shap, "duration", "purposeused_car")


##################################################################################################################

sv_dependence(combined_shap, "duration")
