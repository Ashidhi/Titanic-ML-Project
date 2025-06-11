# Titanic Survival Prediction 🚢

- Improved Random Forest Model with Feature Engineering and Hyperparameter Tuning

- This submission uses a Random Forest Classifier trained on an engineered feature set that includes:

- FamilySize, IsAlone to capture family-related survival patterns.

- Extracted Title from the Name field for social status.

- One-hot encoded features for categorical variables (Sex, Embarked, Title).

- Missing values were handled with:

- Median imputation for Age and Fare

- Mode imputation for Embarked

- A GridSearchCV with 5-fold cross-validation was used for hyperparameter tuning to find the best Random Forest configuration.

- This project predicts passenger survival on the Titanic using machine learning.

## Key Features
- Data cleaning (handling missing values in Age, Fare, Embarked)
- Feature engineering (FamilySize, IsAlone, Title extraction)
- One-hot encoding for categorical variables
- Model comparison: Random Forest vs Logistic Regression
- Hyperparameter tuning using GridSearchCV
- Final accuracy: **0.76555** on Kaggle leaderboard


