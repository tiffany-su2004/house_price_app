from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load("house_price_pipeline.pkl")

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Construct the input dataframe with all required fields
        input_data = pd.DataFrame([{
            'bedrooms': int(data['bedrooms']),
            'bathrooms': float(data['bathrooms']),
            'sqft_living': int(data['sqft_living']),
            'sqft_lot': int(data['sqft_lot']),
            'floors': float(data['floors']),
            'waterfront': int(data['waterfront']),
            'view': int(data['view']),
            'condition': int(data['condition']),
            'sqft_above': int(data['sqft_above']),
            'sqft_basement': int(data['sqft_basement']),
            'city': data['city'],
            'statezip': data['statezip'],
            'total_sqft': int(data['total_sqft']),
            'age': int(data['age']),
            'was_renovated': int(data['was_renovated']),
            'year_sold': int(data['year_sold'])
        }])

        prediction = pipeline.predict(input_data)[0]
        prediction = f"${prediction:,.2f}"

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Something went wrong: {e}")

        return f"Something went wrong: {e}"

if __name__ == "__main__":
    app.run(debug=True)
