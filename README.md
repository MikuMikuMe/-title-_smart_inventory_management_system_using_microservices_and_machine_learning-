# **Title: Smart Inventory Management System Using Microservices and Machine Learning**

```python
# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Load sample inventory data
inventory_data = pd.read_csv('inventory_data.csv')

# Define machine learning model for demand forecasting
def train_model(data):
    X = data[['product_price', 'product_category_id']]
    y = data['demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

    return model

model = train_model(inventory_data)

# Define API route for demand forecasting
@app.route('/forecast_demand', methods=['POST'])
def forecast_demand():
    data = request.get_json()

    product_price = data['product_price']
    product_category_id = data['product_category_id']

    demand_forecast = model.predict([[product_price, product_category_id]])

    return jsonify({'demand_forecast': demand_forecast})

# Error handling for invalid requests
@app.errorhandler(400)
def error_400(e):
    return jsonify({'message': 'Bad Request'}), 400

@app.errorhandler(404)
def error_404(e):
    return jsonify({'message': 'Resource Not Found'}), 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
```

This is a basic implementation of a Smart Inventory Management System using microservices and machine learning. You may need to customize and expand upon this code based on your specific requirements and data structures. Make sure to include sample inventory data in a CSV file and update the data columns in the `train_model` function according to your dataset.