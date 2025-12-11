from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        store_id = float(request.form['store_id'])
        month = float(request.form['month'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        holiday_flag = float(request.form['holiday_flag'])
        prev_month_sales = float(request.form['prev_month_sales'])
        prev_2_months_sales = float(request.form['prev_2_months_sales'])
        prev_year_sales = float(request.form['prev_year_sales'])
        
        # Create feature array
        features = np.array([[store_id, month, temperature, fuel_price, cpi,
                              unemployment, holiday_flag, prev_month_sales,
                              prev_2_months_sales, prev_year_sales]])
        
        # Predict
        prediction = model.predict(features)[0]
        
        return render_template('index.html', prediction_text=f"Next Month Sales: {prediction:.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
