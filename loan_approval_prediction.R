# instaling modules

install.packages(c("tidyverse", "dplyr", "tidyr"))
install.packages("caret")
install.packages("tm")

# Loading libraries
library(caret)
library(tidyverse)
library(tm)
library(dplyr)
library(tidyr)
library(randomForest)
library(glmnet)
library(xgboost)
library(e1071)

# Data cleaning

# loading csv loan datset
loan_data <- read.csv("dataset.csv")
head(loan_data)
tail(loan_data)

# Data Cleaning and Analysis

# Identify missing values
missing_values <- loan_data %>% summarise_all(~sum(is.na(.)))

# View columns with missing values
print(missing_values)

# view the structure of the data
str(loan_data)

# Display summary statistics
summary(loan_data)

# number of features (attributes)
num_features <- length(colnames(loan_data))
cat("The number of features (attributes) are : ", num_features, "\n")

# list the name of instances
list_features <- colnames(loan_data)
cat("\n The list of attributes are: \n")
list_features

# Univariate Analysis
# Histogram of loan_amount
ggplot(loan_data, aes(x = loan_amount)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of Loan Amount", x = "Loan Amount", y = "Frequency")


# Histogram of annual income
ggplot(loan_data, aes(x = income_annum)) +
  geom_histogram(fill = "green", color = "black") +
  labs(title = "Histogram of Annual Income", x = "Annual Income", y = "Frequency")


# Plotting the frequency of education
ggplot(data = loan_data, aes(x = education)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Frequency of Education",
       x = "Education",
       y = "Frequency")


# Plotting the frequency of self_employed
ggplot(data = loan_data, aes(x = self_employed)) +
  geom_bar(fill = "lightgreen", color = "black") +
  labs(title = "Frequency of Self Employment",
       x = "Self Employed",
       y = "Frequency")

# Scatterplot of loan_amount vs. income_annum
ggplot(loan_data, aes(x = income_annum, y = loan_amount)) +
  geom_point(color = "blue") +
  labs(title = "Scatterplot of Loan Amount vs. Annual Income", x = "Annual Income", y = "Loan Amount")

# Grouped bar plot of loan_status by education
loan_data <- mutate(loan_data, education = factor(education, levels = c("Not Graduate", "Graduate")))
ggplot(loan_data, aes(x = education, fill = loan_status)) +
  geom_bar(position = "dodge", color = "black") +
  labs(title = "Loan Status by Education Level", x = "Education Level", y = "Count") +
  scale_fill_manual(values = c("Approved" = "green", "Rejected" = "red"))

# Plotting the frequency of self_employed
ggplot(data = loan_data, aes(x = self_employed)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Frequency of Self Employment",
       x = "Self Employed",
       y = "Frequency")

# Covert categorical variables into integers
loan_data$education <- as.integer(loan_data$education)
loan_data$self_employed <- as.integer(loan_data$self_employed)
loan_data$loan_status <- as.integer(loan_data$loan_status)

# Covert factor levels to integers by converting back to factors
loan_data$education <- as.factor(loan_data$education)
loan_data$self_employed <- as.factor(loan_data$self_employed)
loan_data$loan_status <- as.factor(loan_data$loan_status)

head(loan_data)

# Split the data into 80% training and 20% testing
set.seed(123) # Set seed for reproducibility
train_set <- sample(1:nrow(loan_data), 0.8 * nrow(loan_data))

train <- loan_data[train_set, ]
test <- loan_data[-train_set, ]

# Dimensions of train data and test data
cat("dimension of train data is: ")
cat(dim(train))
cat("\n dimension of test data is:  ")
cat(dim(test))

str(loan_data)

# Load required libraries
library(dplyr)

# Feature Standardization
# Define function for standardization (Z-score scaling)
standardize <- function(x) {
  (x - mean(x)) / sd(x)
}

# Apply standardization to selected numerical variables
loan_data_standardized <- loan_data %>%
  mutate_at(vars(income_annum, loan_amount, cibil_score,
                 residential_assets_value, commercial_assets_value,
                 luxury_assets_value, bank_asset_value),
            standardize)

# Display the updated dataset
head(loan_data_standardized)
tail(loan_data_standardized)

# Train logistic regression model

logistic_model <- train(loan_status ~ ., data = train, method = "glm",
                        family = "binomial")
# Train decision tree model
tree_model <- train(loan_status ~ ., data = train, method = "rpart")

# Train random forest model
rf_model <- train(loan_status ~ ., data = train, method = "rf")

# Train SVM model
svm_model <- svm(loan_status ~ ., data = train, kernel = "radial")

# logistic regression model performance
print(logistic_model)

# Tree model performance
print(tree_model)

# Random model performance
print(rf_model)

# SVM model performance
print(svm_model)

# Predictions of each algorithms for new data (test data)
logistic_preds <- predict(logistic_model, newdata = test)
tree_preds <- predict(tree_model, newdata = test)
rf_preds <- predict(rf_model, newdata = test)
svm_preds <- predict(svm_model, newdata = test)

# Evaluate logistic regression model
confusionMatrix(logistic_preds, test$loan_status)

# Evaluate decision tree model
confusionMatrix(tree_preds, test$loan_status)

# Evaluate random forest model
confusionMatrix(rf_preds, test$loan_status)

# Evaluate SVM model
confusionMatrix(svm_preds, test$loan_status)

# Load required libraries
library(caret)
library(ggplot2)

# Model names
model_names <- c("Logistic Regression", "Decision Tree", "Random Forest", "SVM")

# Model predictions
predictions <- list(logistic_preds, tree_preds, rf_preds, svm_preds)

# Model accuracies
accuracies <- sapply(predictions, function(preds) {
  confusionMatrix(preds, test$loan_status)$overall["Accuracy"]
})

# Create dataframe
model_accuracy_df <- data.frame(Model = model_names, Accuracy = accuracies)

# Plot
ggplot(model_accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Load the necessary library
library(psych)

# Plot the scatter plots and correlations
pairs.panels(loan_data)

