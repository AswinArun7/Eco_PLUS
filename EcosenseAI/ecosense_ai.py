import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Generate Fake Sensor Data (Based on 5 Features)
np.random.seed(42)
data_size = 1000

data = pd.DataFrame({
    'Temperature': np.random.uniform(20, 100, data_size),     # °C
    'Humidity': np.random.uniform(30, 90, data_size),         # %
    'SmokeValue': np.random.uniform(100, 1000, data_size),    # Smoke sensor raw value
    'OilLevel': np.random.uniform(300, 1000, data_size),      # mL
    'VibrationValue': np.random.uniform(50, 300, data_size),  # Vibration sensor
    'Component_Status': np.random.choice([0, 1], data_size)   # 0 = Good, 1 = At Risk
})

# Step 2: Prepare Data
X = data.drop(columns=['Component_Status'])
y = data['Component_Status']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Predict for New Real-Time Sensor Data (Example)
new_data = np.array([[50, 70, 200, 700, 150]])  # Replace with actual values from API
new_data_scaled = scaler.transform(new_data)

new_data_df = pd.DataFrame(new_data_scaled, columns=X.columns)
prediction = model.predict(new_data_df)[0]
status = "At Risk 🚨" if prediction == 1 else "Good ✅"

print(f"Predicted Component Status: {status}")

# Optional: Visualization
plt.figure(figsize=(10, 5))
plt.scatter(data['Temperature'], data['SmokeValue'], c=data['Component_Status'], cmap='coolwarm', alpha=0.5)
plt.xlabel("Temperature")
plt.ylabel("Smoke Value")
plt.title("Temperature vs Smoke Value (Color = Component Status)")
plt.colorbar(label="Component Status (0=Good, 1=At Risk)")
plt.show()
