

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Upload Titanic dataset files
from google.colab import files

uploaded = files.upload()






# Load CSV files into pandas DataFrames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Display the first 5 rows of the training data
print("Train data sample:")
display(train.head())

# Check basic info and missing values in training set
print("\nTrain data info:")
train.info()

print("\nMissing values in train data:")
print(train.isnull().sum())







# Fill missing Age with median age
median_age = train['Age'].median()
train['Age'].fillna(median_age, inplace=True)

# Fill missing Embarked with mode (most frequent port)
mode_embarked = train['Embarked'].mode()[0]
train['Embarked'].fillna(mode_embarked, inplace=True)

# Drop Cabin column (too many missing values)
train.drop('Cabin', axis=1, inplace=True)

# Check again for missing values after cleaning
print(train.isnull().sum())







# Survival count plot
sns.countplot(data=train, x='Survived')
plt.title('Survival Count')
plt.show()

# Survival by Sex
sns.countplot(data=train, x='Survived', hue='Sex')
plt.title('Survival Count by Sex')
plt.show()

# Survival by Pclass
sns.countplot(data=train, x='Survived', hue='Pclass')
plt.title('Survival Count by Passenger Class')
plt.show()

# Create a Family Size feature (SibSp + Parch + 1)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

# Survival by Family Size
sns.countplot(data=train, x='Survived', hue='FamilySize')
plt.title('Survival Count by Family Size')
plt.show()

# Age distribution by survival
sns.histplot(data=train, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.show()







# Convert categorical columns to numeric using one-hot encoding
train_encoded = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target
features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_encoded[features]
y = train_encoded['Survived']

# Split data into training and validation sets (80% train, 20% validation)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training samples: {X_train.shape[0]}')
print(f'Validation samples: {X_val.shape[0]}')






#Here’s the code to train a Random Forest classifier and check its accuracy:

# Create the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on validation data
y_pred = model.predict(X_val)

# Check accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f' Random Forest Accuracy:{accuracy:.2f}')






from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print(classification_report(y_val, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Survived'], yticklabels=['Dead', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()







from sklearn.model_selection import GridSearchCV

# Define parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [None, 5, 10, 20],        # Max depth of each tree
    'min_samples_split': [2, 5, 10],       # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4]          # Minimum samples at leaf node
}

# Create a RandomForest model
rf = RandomForestClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)

# Fit on training data
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Use best estimator to predict
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_val)

# Accuracy with best model
from sklearn.metrics import accuracy_score
print("Validation accuracy after tuning:", accuracy_score(y_val, y_pred_best))









# Get feature importances from the best model
importances = best_model.feature_importances_

# Match to column names
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()










from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Create and train model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict
logreg_preds = logreg.predict(X_val)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_val, logreg_preds))

print("\nClassification Report:")
print(classification_report(y_val, logreg_preds))

# Cross-validation score
logreg_cv_scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("\nCross-validation Accuracy Scores:", logreg_cv_scores)
print("Average CV Accuracy:", logreg_cv_scores.mean())








coeff_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

coeff_df






# 1. Accuracy comparison


print("Logistic Regression Accuracy:", logreg.score(X_val, y_val))
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred))

#2. Classification reports side-by-side"""

print("Logistic Regression Classification Report:\n", classification_report(y_val, logreg.predict(X_val)))
print("Random Forest Classification Report:\n", classification_report(y_val,best_model.predict(X_val) ))

#3. Confusion Matrices"""

fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_val, logreg.predict(X_val)), annot=True, fmt='d', ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")

sns.heatmap(confusion_matrix(y_val, best_model.predict(X_val)), annot=True, fmt='d', ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix")

plt.show()










##Clean and preprocess the test dataset just like the training set


# Load the test data

test_df = pd.read_csv('test.csv')
test_df.head()

#check for missing values in test data
print(test_df.isnull().sum())

# Fill Age
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Fill Fare
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop Cabin (too many missing)
test_df.drop('Cabin', axis=1, inplace=True)

# If Embarked missing, fill with mode (usually no missing in test)
if test_df['Embarked'].isnull().sum() > 0:
    test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

#check for missing values in test data
print(test_df.isnull().sum())

# Feature engineering & encoding (same as training)

# Example family size
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# Encode Sex: male=1, female=0 (assuming you did label encoding)
test_df['Sex_male'] = test_df['Sex'].map({'male': 1, 'female': 0})

# One-hot encode Embarked
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Prepare features for prediction

# Select the columns your model was trained on:
features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X_test = test_df[features]

# Predict survival with Random Forest
test_preds = best_model.predict(X_test)

# Create submission file

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_preds
})

submission.to_csv('submission.csv', index=False)

# download the file:
from google.colab import files
files.download('submission.csv')

# To preview the content of the submission file in notebook:

submission.head()

