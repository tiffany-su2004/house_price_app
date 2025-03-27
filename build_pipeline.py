import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cleaned_houseprice_dataset.csv")

# Define features
X = df.drop("price", axis=1)
y = df["price"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = ["city", "statezip"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Final pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save it — this file is what app.py will use
joblib.dump(pipeline, "house_price_pipeline.pkl")

print("✅ Pipeline trained and saved successfully.")
