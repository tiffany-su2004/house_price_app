# test_accuracy.py
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Load your pipeline and test data
pipeline = joblib.load("house_price_pipeline.pkl")
test_data = pd.read_csv("cleaned_houseprice_dataset.csv")  # this should include actual prices

# Separate features from target
X_test = test_data.drop(columns=['price'])
y_test = test_data['price']

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate accuracy
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.2%}")
