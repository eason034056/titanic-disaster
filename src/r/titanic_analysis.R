library(readr)   
library(dplyr) 
library(tidyr)   
library(caret) 

cat("================================================================================\n")
cat("Titanic Survival Prediction Analysis (R Implementation)\n")
cat("================================================================================\n\n")

# Step 1: Load training data
cat("Step 1 (Q14): Loading training data...\n")
cat("--------------------------------------------------------------------------------\n")

# Read CSV file
train_df <- read_csv("data/train.csv", show_col_types = FALSE)

cat("Training data loaded successfully!\n")
cat(sprintf("Dataset dimensions: %d rows × %d columns\n", nrow(train_df), ncol(train_df)))
cat("\n") 

# Display first few rows of data
cat("First 5 rows of the dataset:\n")
print(head(train_df, 5))
cat("\n")

# Display data structure
cat("Dataset structure:\n")
str(train_df)
cat("\n")

# Step 2: Explore data
cat("\nStep 2 (Q14): Exploratory Data Analysis (EDA)\n")
cat("--------------------------------------------------------------------------------\n")

# Check for missing values
cat("Missing values in each column:\n")
missing_values <- sapply(train_df, function(x) sum(is.na(x)))
print(missing_values)
cat("\n")

# Display survival rate statistics
cat("Survival statistics:\n")
survival_rate <- mean(train_df$Survived)
survived_count <- sum(train_df$Survived == 1)
not_survived_count <- sum(train_df$Survived == 0)
cat(sprintf("Overall survival rate: %.2f%%\n", survival_rate * 100))
cat(sprintf("Survived: %d passengers\n", survived_count))
cat(sprintf("Did not survive: %d passengers\n", not_survived_count))
cat("\n")

# Survival rate by gender
cat("Survival rate by gender:\n")
gender_survival <- train_df %>%
  group_by(Sex) %>%
  summarise(
    survival_rate = mean(Survived),
    count = n()
  )
print(gender_survival)
cat("\n")

# Survival rate by passenger class
cat("Survival rate by passenger class:\n")
class_survival <- train_df %>%
  group_by(Pclass) %>%
  summarise(
    survival_rate = mean(Survived),
    count = n()
  )
print(class_survival)
cat("\n")

# Step 3: Data preprocessing
cat("\nStep 3 (Q14): Data Preprocessing\n")
cat("--------------------------------------------------------------------------------\n")

# Create a copy of the data
df <- train_df

cat("Handling missing values...\n")

# Age: Fill with median
age_median <- median(df$Age, na.rm = TRUE)
df$Age[is.na(df$Age)] <- age_median
cat(sprintf("  - Filled missing Age values with median: %.2f\n", age_median))

# Embarked: Fill with most common value
embarked_mode <- names(sort(table(df$Embarked), decreasing = TRUE))[1]
df$Embarked[is.na(df$Embarked)] <- embarked_mode
cat(sprintf("  - Filled missing Embarked values with mode: %s\n", embarked_mode))

# Cabin: Create binary feature
df$HasCabin <- as.integer(!is.na(df$Cabin))
cat("  - Created HasCabin feature (1 if cabin number exists, 0 otherwise)\n")
cat("\n")

# Feature engineering
cat("Feature engineering...\n")

# Convert gender to numeric
df$Sex_binary <- as.integer(df$Sex == "female")
cat("  - Converted Sex to binary: male=0, female=1\n")

# Convert Embarked to numeric
df$Embarked_numeric <- case_when(
  df$Embarked == "S" ~ 0,
  df$Embarked == "C" ~ 1,
  df$Embarked == "Q" ~ 2,
  TRUE ~ 0
)
cat("  - Converted Embarked to numeric: S=0, C=1, Q=2\n")

# Create family size feature
df$FamilySize <- df$SibSp + df$Parch + 1
cat("  - Created FamilySize feature: SibSp + Parch + 1\n")

# Create traveling alone feature
df$IsAlone <- as.integer(df$FamilySize == 1)
cat("  - Created IsAlone feature: 1 if traveling alone, 0 otherwise\n")
cat("\n")

# Select features
feature_columns <- c('Pclass', 'Sex_binary', 'Age', 'SibSp', 'Parch', 
                     'Fare', 'Embarked_numeric', 'FamilySize', 'IsAlone', 'HasCabin')

cat("Selected features for the model:\n")
for (i in seq_along(feature_columns)) {
  cat(sprintf("  %d. %s\n", i, feature_columns[i]))
}
cat("\n")

# Prepare feature matrix and target variable
X <- df[, feature_columns]
y <- as.factor(df$Survived)  # Convert to factor (categorical variable)

cat(sprintf("Feature matrix dimensions: %d × %d\n", nrow(X), ncol(X)))
cat(sprintf("Target variable length: %d\n", length(y)))
cat("\n")

# Step 4: Build logistic regression model
cat("\nStep 4 (Q15): Building Logistic Regression Model\n")
cat("--------------------------------------------------------------------------------\n")

# Set training control parameters
train_control <- trainControl(
  method = "none"  # No cross-validation, direct training
)

cat("Training Logistic Regression model...\n")

# Train the model
model <- train(
  x = X,
  y = y,
  method = "glm",              # Generalized Linear Model (logistic regression)
  family = binomial,           # Binary classification
  trControl = train_control
)

cat("Model training completed!\n\n")

# Display model summary
cat("Model summary:\n")
print(summary(model$finalModel))
cat("\n")

# Step 5: Evaluate model on training set
cat("\nStep 5 (Q16): Evaluating Model on Training Set\n")
cat("--------------------------------------------------------------------------------\n")

# Make predictions
train_predictions <- predict(model, X)

# Calculate accuracy
confusion <- confusionMatrix(train_predictions, y)
train_accuracy <- confusion$overall['Accuracy']

cat(sprintf("Training Accuracy: %.4f (%.2f%%)\n\n", train_accuracy, train_accuracy * 100))

# Display confusion matrix
cat("Confusion Matrix (Training Set):\n")
print(confusion$table)
cat("\n")

cat("Detailed Statistics:\n")
print(confusion$byClass)
cat("\n")

# Step 6: Load test data
cat("\nStep 6 (Q17): Loading Test Data\n")
cat("--------------------------------------------------------------------------------\n")

test_df <- read_csv("data/test.csv", show_col_types = FALSE)
cat("Test data loaded successfully!\n")
cat(sprintf("Test dataset dimensions: %d rows × %d columns\n\n", nrow(test_df), ncol(test_df)))

# Step 7: Preprocess test data
cat("Step 7 (Q17): Preprocessing Test Data\n")
cat("--------------------------------------------------------------------------------\n")

# Create a copy of test data
test_processed <- test_df

# Handle missing values
test_processed$Age[is.na(test_processed$Age)] <- age_median
test_processed$Fare[is.na(test_processed$Fare)] <- median(test_processed$Fare, na.rm = TRUE)
test_processed$Embarked[is.na(test_processed$Embarked)] <- embarked_mode

# Feature engineering
test_processed$Sex_binary <- as.integer(test_processed$Sex == "female")
test_processed$Embarked_numeric <- case_when(
  test_processed$Embarked == "S" ~ 0,
  test_processed$Embarked == "C" ~ 1,
  test_processed$Embarked == "Q" ~ 2,
  TRUE ~ 0
)
test_processed$FamilySize <- test_processed$SibSp + test_processed$Parch + 1
test_processed$IsAlone <- as.integer(test_processed$FamilySize == 1)
test_processed$HasCabin <- as.integer(!is.na(test_processed$Cabin))

# Prepare test features
X_test <- test_processed[, feature_columns]

cat("Test data preprocessing completed!\n")
cat(sprintf("Test feature matrix dimensions: %d × %d\n\n", nrow(X_test), ncol(X_test)))

# Step 8: Make predictions on test set
cat("\nStep 8 (Q18): Making Predictions on Test Set\n")
cat("--------------------------------------------------------------------------------\n")

# Make predictions
test_predictions <- predict(model, X_test)
# Convert back to numeric (0 or 1)
test_predictions_numeric <- as.integer(as.character(test_predictions))

cat("Predictions completed!\n")
cat(sprintf("Number of predictions: %d\n", length(test_predictions_numeric)))
cat(sprintf("Predicted survived: %d\n", sum(test_predictions_numeric == 1)))
cat(sprintf("Predicted did not survive: %d\n", sum(test_predictions_numeric == 0)))
cat("\n")

# Display some prediction results
cat("Sample predictions (first 10):\n")
sample_results <- data.frame(
  PassengerId = test_df$PassengerId[1:10],
  Prediction = test_predictions_numeric[1:10],
  Prediction_Label = ifelse(test_predictions_numeric[1:10] == 1, "Survived", "Did not survive")
)
print(sample_results)
cat("\n")

# Step 9: Compare with gender_submission.csv to evaluate accuracy
cat("\nStep 9 (Q18): Comparing Predictions with gender_submission.csv\n")
cat("--------------------------------------------------------------------------------\n")

# Read gender_submission.csv
cat("Loading gender_submission.csv for comparison...\n")
gender_submission <- read_csv("data/gender_submission.csv", show_col_types = FALSE)

# Ensure PassengerId order is consistent
gender_submission <- gender_submission[order(gender_submission$PassengerId), ]
test_ordered <- data.frame(
  PassengerId = test_df$PassengerId,
  Predicted = test_predictions_numeric
)
test_ordered <- test_ordered[order(test_ordered$PassengerId), ]

# Calculate accuracy
matching_predictions <- sum(gender_submission$Survived == test_ordered$Predicted)
total_predictions <- nrow(gender_submission)
accuracy <- matching_predictions / total_predictions

cat(sprintf("Total test samples: %d\n", total_predictions))
cat(sprintf("Matching predictions: %d\n", matching_predictions))
cat(sprintf("Test accuracy: %.2f%%\n\n", accuracy * 100))

# Display confusion matrix
confusion <- table(Actual = gender_submission$Survived, Predicted = test_ordered$Predicted)
cat("Confusion Matrix:\n")
print(confusion)

# Calculate other evaluation metrics
true_positives <- confusion[2, 2]
false_positives <- confusion[1, 2]
true_negatives <- confusion[1, 1]
false_negatives <- confusion[2, 1]

precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * precision * recall / (precision + recall)

cat("\nAdditional Metrics:\n")
cat(sprintf("Precision: %.4f\n", precision))
cat(sprintf("Recall: %.4f\n", recall))
cat(sprintf("F1 Score: %.4f\n", f1_score))
cat("\n")

# Save prediction results
submission <- data.frame(
  PassengerId = test_df$PassengerId,
  Survived = test_predictions_numeric
)
write.csv(submission, "data/submission.csv", row.names = FALSE)


# Completion
cat("================================================================================\n")
cat("Analysis Completed Successfully!\n")
cat("================================================================================\n\n")
