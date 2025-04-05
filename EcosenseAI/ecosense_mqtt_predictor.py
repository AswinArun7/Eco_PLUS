import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import requests

# ---------- STEP 1: Load and Clean Dataset ----------
def load_and_clean_dataset():
    try:
        # Look for the dataset in several possible locations
        possible_paths = [
            "ecosense_dataset.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecosense_dataset.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ecosense_dataset.csv"),
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path:
            print(f"\n📥 Found dataset at: {dataset_path}")
            raw_df = pd.read_csv(dataset_path)
            
            if raw_df.shape[1] == 1:
                df = raw_df.iloc[:, 0].str.split(r'\s+', expand=True)
                df.columns = [
                    'Timestamp', 'Temperature', 'Humidity', 'SmokeValue', 'OilLevel', 'VibrationValue',
                    'Smoke_Label', 'Air_Label', 'Battery_Label', 'Engine_Label'
                ]
            else:
                df = raw_df.copy()

            numeric_fields = ['Temperature', 'Humidity', 'SmokeValue', 'OilLevel', 'VibrationValue']
            label_fields = ['Smoke_Label', 'Air_Label', 'Battery_Label', 'Engine_Label']
            df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric)
            df[label_fields] = df[label_fields].apply(pd.to_numeric)

            print("\n📥 Dataset loaded and cleaned successfully!")
            return df
        else:
            print("\n⚠️ Dataset file not found. Creating mock dataset.")
            return create_mock_dataset()
    except Exception as e:
        print(f"\n❌ Error loading or processing dataset: {str(e)}")
        print("Creating mock dataset instead.")
        return create_mock_dataset()

def create_mock_dataset():
    # Create a mock dataset for demonstration purposes
    print("\n📝 Creating mock dataset for demonstration")
    
    # Generate timestamps for the last 100 days
    timestamps = [
        (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(100)
    ]
    
    # Generate random sensor data
    np.random.seed(42)  # For reproducibility
    temperature = np.random.uniform(60, 100, 100)
    humidity = np.random.uniform(30, 80, 100)
    smoke_value = np.random.uniform(10, 50, 100)
    oil_level = np.random.uniform(40, 90, 100)
    vibration_value = np.random.uniform(20, 70, 100)
    
    # Generate labels based on thresholds
    smoke_label = (smoke_value > 40).astype(int) * 2 + (smoke_value > 30).astype(int) * (smoke_value <= 40).astype(int)
    air_label = (humidity > 70).astype(int) * 2 + (humidity > 60).astype(int) * (humidity <= 70).astype(int)
    battery_label = (vibration_value > 60).astype(int) * 2 + (vibration_value > 50).astype(int) * (vibration_value <= 60).astype(int)
    engine_label = ((oil_level < 50) | (temperature > 90) | (vibration_value > 60)).astype(int) * 2 + \
                   ((oil_level < 60) | (temperature > 80) | (vibration_value > 50)).astype(int) * \
                   (~((oil_level < 50) | (temperature > 90) | (vibration_value > 60))).astype(int)
    
    # Create a DataFrame
    data = {
        'Timestamp': timestamps,
        'Temperature': temperature,
        'Humidity': humidity,
        'SmokeValue': smoke_value,
        'OilLevel': oil_level,
        'VibrationValue': vibration_value,
        'Smoke_Label': smoke_label,
        'Air_Label': air_label,
        'Battery_Label': battery_label,
        'Engine_Label': engine_label
    }
    
    df = pd.DataFrame(data)
    print("\n✅ Mock dataset created successfully with 100 rows")
    
    return df

# ---------- STEP 2: Train AI Models ----------
def train_model(df, feature, label):
    X = df[[feature]] if isinstance(feature, str) else df[feature]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f"✅ {label.replace('_Label', '')} Model Accuracy: {acc:.2f}%")
    return model

def train_all_models(df):
    models = {}
    models['Smoke'] = train_model(df, 'SmokeValue', 'Smoke_Label')
    models['Air'] = train_model(df, 'Humidity', 'Air_Label')
    models['Battery'] = train_model(df, 'VibrationValue', 'Battery_Label')
    models['Engine'] = train_model(df, ['OilLevel', 'Temperature', 'VibrationValue'], 'Engine_Label')
    return models

# ---------- STEP 3: Fetch Real-Time Sensor Data ----------
def fetch_sensor_data():
    url = "https://65e4-117-239-78-56.ngrok-free.app/arduino-data"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            sensor = response.json().get('data', {})
            keys = ['Temperature', 'Humidity', 'SmokeValue', 'OilLevel', 'VibrationValue']
            if not all(k in sensor for k in keys):
                print(f"\n❌ Missing keys in sensor data: {[k for k in keys if k not in sensor]}")
                return None
            print("\n📡 Real-Time Sensor Data:")
            for key in keys:
                print(f" - {key}: {sensor[key]}")
            return sensor
        else:
            print(f"\n❌ Failed to fetch data. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"\n❌ Exception occurred while fetching sensor data: {str(e)}")
        return None

# ---------- STEP 4: Make Predictions ----------
def make_predictions(sensor_data, models):
    label_map = {
        0: "🟢 GOOD – No issue expected for 3+ months",
        1: "🟡 WARNING – May fail in 1–3 months",
        2: "🔴 CRITICAL – May fail within 1 week"
    }

    predictions = {
        'Smoke': models['Smoke'].predict([[sensor_data['SmokeValue']]])[0],
        'Air': models['Air'].predict([[sensor_data['Humidity']]])[0],
        'Battery': models['Battery'].predict([[sensor_data['VibrationValue']]])[0],
        'Engine': models['Engine'].predict([[sensor_data['OilLevel'], sensor_data['Temperature'], sensor_data['VibrationValue']]])[0]
    }

    predicted_health = {
        comp: label_map.get(pred, 'Unknown') for comp, pred in predictions.items()
    }

    print("\n🔍 Predicted Component Health:")
    for comp, pred in predicted_health.items():
        print(f" - {comp}: {pred}")

    return predicted_health

# ---------- STEP 5: Log Predictions ----------
def log_predictions(predictions):
    log_file = "predictions_log.csv"
    log_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Smoke': predictions['Smoke'],
        'Air': predictions['Air'],
        'Battery': predictions['Battery'],
        'Engine': predictions['Engine']
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    print("\n📝 Prediction logged successfully!")

# ---------- STEP 6: Generate and Plot Graphs ----------
def generate_and_plot_graphs(predictions):
    def generate_component_data(length=29, start_val=100, end_val=70, noise_scale=1.5):
        x = np.linspace(0, 4 * np.pi, length)
        base_line = np.linspace(start_val, end_val, length)
        wave = 4 * np.sin(x)
        noise = np.random.normal(scale=noise_scale, size=length)
        return base_line + wave + noise

    efficiency_mapping = {
        0: 100,
        1: 75,
        2: 40
    }

    components = {
        "Smoke Sensor Efficiency": ("Smoke", "#00FF99"),
        "Air Quality Efficiency": ("Air", "#00BFFF"),
        "Battery Health Efficiency": ("Battery", "#FF33CC"),
        "Engine Health Efficiency": ("Engine", "#FFA500")
    }

    plt.style.use('dark_background')

    for title, (comp_key, color) in components.items():
        y_data = list(generate_component_data())
        real_eff = efficiency_mapping.get(predictions[comp_key], 0)
        y_data.append(real_eff)

        x_data = np.arange(len(y_data))

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#111')
        ax.set_facecolor('#111')

        ax.plot(x_data, y_data, color=color, linewidth=2.5, marker='o', markersize=6,
                markerfacecolor='white', markeredgewidth=1.2)

        ax.plot(x_data[-1], y_data[-1], marker='x', color='red', markersize=10, mew=3,
                label=f"Current: {y_data[-1]:.1f}%")

        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel("Time", color='gray')
        ax.set_ylabel("Efficiency (%)", color='gray')
        ax.set_ylim(0, 120)
        ax.tick_params(colors='gray')
        ax.grid(True, linestyle='--', alpha=0.3)

        legend = ax.legend(loc="upper right", frameon=True, facecolor="#222", edgecolor="#444")
        for text in legend.get_texts():
            text.set_color('white')

        plt.show()

# ---------- Main Function: Flask integration ----------
def run_prediction(component_type='engine'):
    """
    Run prediction for a specific component type.
    
    Args:
        component_type (str): The component to focus on ('engine', 'air', 'battery', 'smoke').
                             Default is 'engine'.
    
    Returns:
        dict: Prediction results for all components.
    """
    print(f"Running prediction for component: {component_type}")
    
    try:
        # Load and clean dataset
        df = load_and_clean_dataset()
        if df is None:
            return {"error": "Dataset loading failed"}

        # Train models
        models = train_all_models(df)

        # For testing purposes, create a mock sensor data if API call fails
        try:
            # Fetch real-time sensor data
            sensor_data = fetch_sensor_data()
            if sensor_data is None:
                # Create mock data as fallback
                sensor_data = {
                    'Temperature': 75,
                    'Humidity': 65,
                    'SmokeValue': 30,
                    'OilLevel': 60,
                    'VibrationValue': 45
                }
                print("Using mock sensor data since API call failed")
        except Exception as e:
            # Create mock data on exception
            sensor_data = {
                'Temperature': 75,
                'Humidity': 65,
                'SmokeValue': 30,
                'OilLevel': 60,
                'VibrationValue': 45
            }
            print(f"Error fetching sensor data: {e}. Using mock data instead.")

        # Make predictions
        predictions = make_predictions(sensor_data, models)

        # Log predictions
        try:
            log_predictions(predictions)
        except Exception as e:
            print(f"Error logging predictions: {e}")

        # Skip generating graphs when running as API
        # generate_and_plot_graphs(predictions)

        return predictions
    except Exception as e:
        print(f"Error in run_prediction: {e}")
        return {"error": str(e)}

