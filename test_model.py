import joblib
import pandas as pd

# Load model
pipeline = joblib.load("house_price_pipeline.pkl")

# Sample data (must match the columns during training)
data = pd.DataFrame([{
    'bedrooms': 3,
    'bathrooms': 2.0,
    'sqft_living': 1800,
    'sqft_lot': 5000,
    'floors': 1.0,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'sqft_above': 1800,
    'sqft_basement': 0,
    'city': 'Seattle',
    'statezip': 'WA 98103',
    'total_sqft': 1800,
    'age': 35,
    'was_renovated': 0,
    'year_sold': 2025
}])

# Predict
prediction = pipeline.predict(data)
print(f"Predicted Price: ${prediction[0]:,.2f}")
