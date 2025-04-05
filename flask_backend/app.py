from flask import Flask, jsonify, request, make_response
import sys
import os
import datetime

# Add EcosenseAI folder to the system path dynamically using the absolute path
ecosense_ai_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'EcosenseAI')
sys.path.append(ecosense_ai_path)  # Modify the path to the correct folder name

# Import your AI script (modify function names if different)
from ecosense_mqtt_predictor import run_prediction  # Assuming this function runs your model

app = Flask(__name__)

# Handle CORS manually without flask_cors
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/')
def index():
    return "Welcome to the Ecosense AI Flask App!"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve query parameter 'component' (default is 'engine' if not provided)
        component_type = request.args.get('component', default='engine')  # Default is 'engine'
        print(f"Predicting for component: {component_type}")

        # Call the AI model's function to make predictions with the provided component type
        prediction = run_prediction(component_type)  # Assuming your AI function can take this parameter

        # Format the prediction data with more details
        response_data = {
            "prediction": prediction,
            "status": "success",
            "message": f"Prediction made for {component_type}",
            "timestamp": str(datetime.datetime.now())  # Add a timestamp to the prediction
        }

        return jsonify(response_data)  # Return prediction as a detailed JSON response

    except Exception as e:
        # Handle any errors gracefully and send a meaningful response
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "timestamp": str(datetime.datetime.now())
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
