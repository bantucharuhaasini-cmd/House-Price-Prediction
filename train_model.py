import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & Model Selection
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Model
from sklearn.ensemble import GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Save model
import joblib


# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("data/Housing.csv")


# -------------------------------
# 2. Define Features & Target
# -------------------------------
X = df.drop("price", axis=1)
y = df["price"]


# -------------------------------
# 3. Column Types
# -------------------------------
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

categorical_cols = ['furnishingstatus']


# -------------------------------
# 4. Preprocessing Pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('bin', OrdinalEncoder(), binary_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)


# -------------------------------
# 5. Model Pipeline
# -------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])


# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 7. Train Model
# -------------------------------
model.fit(X_train, y_train)


# -------------------------------
# 8. Predictions
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# 9. Evaluation
# -------------------------------
print("📊 Model Performance:")
print(f"MAE  : {mean_absolute_error(y_test, y_pred)}")
print(f"MSE  : {mean_squared_error(y_test, y_pred)}")
print(f"R2   : {r2_score(y_test, y_pred)}")


# -------------------------------
# 10. Save Pipeline Model
# -------------------------------
joblib.dump(model, "model/house_price_pipeline.pkl")

print("\n✅ Model Pipeline Saved Successfully!")