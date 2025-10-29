import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Titanic Survival Prediction Analysis")
print("="*80)
print()

# Step 1: Load training data
print("Step 1 (Q14): Loading training data...")
print("-" * 80)

# Read CSV file
train_df = pd.read_csv('data/train.csv')

# Display basic information about the data
print(f"Training data loaded successfully!")
print(f"Dataset shape: {train_df.shape}") 
print(f"Number of rows: {train_df.shape[0]}")
print(f"Number of columns: {train_df.shape[1]}")
print()

# Display the first few rows of data
print("First 5 rows of the dataset:")
print(train_df.head())
print()

# Display column information
print("Dataset information:")
print(train_df.info())
print()

# Step 2: Explore data
print("\nStep 2 (Q14): Exploratory Data Analysis (EDA)")
print("-" * 80)

# Check for missing values
print("Missing values in each column:")
print(train_df.isnull().sum())
print()

# Display survival rate statistics
print("Survival statistics:")
survival_rate = train_df['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.2%}")
print(f"Survived: {train_df['Survived'].sum()} passengers")
print(f"Did not survive: {len(train_df) - train_df['Survived'].sum()} passengers")
print()

# Survival rate by gender
print("Survival rate by gender:")
gender_survival = train_df.groupby('Sex')['Survived'].agg(['mean', 'count'])
print(gender_survival)
print()

# Survival rate by passenger class
print("Survival rate by passenger class:")
class_survival = train_df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
print(class_survival)
print()

# Step 3: Data preprocessing
print("\nStep 3 (Q14): Data Preprocessing")
print("-" * 80)

# Create a copy of the data to avoid modifying the original
df = train_df.copy()

# Handle missing values
print("Handling missing values...")

# Age: Fill with median
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
print(f"  - Filled missing Age values with median: {age_median}")

# Embarked: Fill with most common value
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
print(f"  - Filled missing Embarked values with mode: {embarked_mode}")

# Cabin: Too many missing, create binary feature (has/doesn't have)
df['HasCabin'] = df['Cabin'].notna().astype(int)
print(f"  - Created HasCabin feature (1 if cabin number exists, 0 otherwise)")

print()

# Feature engineering
print("Feature engineering...")

# Convert gender to numeric
df['Sex_binary'] = df['Sex'].map({'male': 0, 'female': 1})
print("  - Converted Sex to binary: male=0, female=1")

# Convert Embarked to numeric
df['Embarked_numeric'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print("  - Converted Embarked to numeric: S=0, C=1, Q=2")

# Create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("  - Created FamilySize feature: SibSp + Parch + 1")

# Create is alone feature
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("  - Created IsAlone feature: 1 if traveling alone, 0 otherwise")

print()

# Select features to use
feature_columns = ['Pclass', 'Sex_binary', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'Embarked_numeric', 'FamilySize', 'IsAlone', 'HasCabin']

print("Selected features for the model:")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")
print()

# Prepare feature matrix X and target variable y
X = df[feature_columns]
y = df['Survived']

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print()

# Step 4: Build logistic regression model
print("\nStep 4 (Q15): Building Logistic Regression Model")
print("-" * 80)

# Create logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
print("Created Logistic Regression model with parameters:")
print(f"  - max_iter=1000 (maximum number of iterations)")
print(f"  - random_state=42 (for reproducibility)")
print()

# Train the model
print("Training the model...")
model.fit(X, y)
print("Model training completed!")
print()

# Display feature importance (coefficients)
print("Feature coefficients:")
for feature, coef in zip(feature_columns, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")
print()

# Step 5: Evaluate model on training set
print("\nStep 5 (Q16): Evaluating Model on Training Set")
print("-" * 80)

# Make predictions on training set
train_predictions = model.predict(X)

# Calculate accuracy
train_accuracy = accuracy_score(y, train_predictions)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print()

# Display confusion matrix
print("Confusion Matrix (Training Set):")
conf_matrix = confusion_matrix(y, train_predictions)
print(conf_matrix)
print()
print("Explanation:")
print(f"  True Negatives (TN): {conf_matrix[0][0]} - Correctly predicted did not survive")
print(f"  False Positives (FP): {conf_matrix[0][1]} - Incorrectly predicted survived")
print(f"  False Negatives (FN): {conf_matrix[1][0]} - Incorrectly predicted did not survive")
print(f"  True Positives (TP): {conf_matrix[1][1]} - Correctly predicted survived")
print()

# Display classification report
print("Classification Report (Training Set):")
print(classification_report(y, train_predictions, target_names=['Did not survive', 'Survived']))
print()

# Step 6: Load test data
print("\nStep 6 (Q17): Loading Test Data")
print("-" * 80)

test_df = pd.read_csv('data/test.csv')
print(f"Test data loaded successfully!")
print(f"Test dataset shape: {test_df.shape}")
print()

# Step 7: Preprocess test data
print("Step 7 (Q17): Preprocessing Test Data")
print("-" * 80)

# Apply the same preprocessing to test data
test_processed = test_df.copy()

# Handle missing values
test_processed['Age'].fillna(age_median, inplace=True)
test_processed['Fare'].fillna(test_processed['Fare'].median(), inplace=True)
test_processed['Embarked'].fillna(embarked_mode, inplace=True)

# Feature engineering
test_processed['Sex_binary'] = test_processed['Sex'].map({'male': 0, 'female': 1})
test_processed['Embarked_numeric'] = test_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_processed['FamilySize'] = test_processed['SibSp'] + test_processed['Parch'] + 1
test_processed['IsAlone'] = (test_processed['FamilySize'] == 1).astype(int)
test_processed['HasCabin'] = test_processed['Cabin'].notna().astype(int)

# Prepare test features
X_test = test_processed[feature_columns]

print("Test data preprocessing completed!")
print(f"Test feature matrix shape: {X_test.shape}")
print()

# Step 8: Make predictions on test set
print("\nStep 8 (Q18): Making Predictions on Test Set")
print("-" * 80)

test_predictions = model.predict(X_test)

print(f"Predictions completed!")
print(f"Number of predictions: {len(test_predictions)}")
print(f"Predicted survived: {test_predictions.sum()}")
print(f"Predicted did not survive: {len(test_predictions) - test_predictions.sum()}")
print()

# Display some prediction results
print("Sample predictions (first 10):")
sample_results = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].head(10),
    'Prediction': test_predictions[:10],
    'Prediction_Label': ['Survived' if p == 1 else 'Did not survive' for p in test_predictions[:10]]
})
print(sample_results.to_string(index=False))
print()


# Save prediction results (optional)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# Step 9: Evaluate model on test set by comparing with gender_submission.csv
print("\nStep 9 (Q18): Evaluating Model on Test Set")
print("-" * 80)

# Load the gender_submission.csv file for comparison
try:
    gender_submission = pd.read_csv('data/gender_submission.csv')
    
    # Ensure the PassengerId columns match
    if not all(submission['PassengerId'] == gender_submission['PassengerId']):
        print("Warning: PassengerId columns don't match between prediction and ground truth!")
    else:
        # Calculate accuracy
        correct_predictions = (submission['Survived'] == gender_submission['Survived']).sum()
        total_predictions = len(gender_submission)
        test_accuracy = correct_predictions / total_predictions
        
        print(f"Test set accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Additional metrics
        true_positives = ((submission['Survived'] == 1) & (gender_submission['Survived'] == 1)).sum()
        true_negatives = ((submission['Survived'] == 0) & (gender_submission['Survived'] == 0)).sum()
        false_positives = ((submission['Survived'] == 1) & (gender_submission['Survived'] == 0)).sum()
        false_negatives = ((submission['Survived'] == 0) & (gender_submission['Survived'] == 1)).sum()
        
        print(f"True positives (correctly predicted survivors): {true_positives}")
        print(f"True negatives (correctly predicted non-survivors): {true_negatives}")
        print(f"False positives (incorrectly predicted as survivors): {false_positives}")
        print(f"False negatives (incorrectly predicted as non-survivors): {false_negatives}")
        
except FileNotFoundError:
    print("Warning: Could not find gender_submission.csv for accuracy evaluation.")
    print("Make sure the file exists in the data/ directory.")
except Exception as e:
    print(f"Error evaluating test accuracy: {str(e)}")

print()


submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
print()


# Complete
print("="*80)
print("Analysis Completed Successfully!")
print("="*80)
print()