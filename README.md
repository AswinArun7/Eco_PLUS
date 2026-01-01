
# 🚗 EcoSense

**EcoSense** is an end-to-end **IoT and Machine Learning–based predictive maintenance system** for monitoring vehicle health and detecting potential failures early.

---

## 🔍 Problem

Vehicle maintenance is mostly reactive. Issues are detected only after breakdowns occur, and drivers or fleet operators lack real-time visibility into engine condition, battery health, emissions, and overall performance. Infrequent checks allow minor faults to escalate into major failures.

---

## 💡 Solution

EcoSense continuously collects vehicle sensor data, visualizes live telemetry in a mobile app, and uses machine learning to classify component health as **Good**, **Warning**, or **Critical**, enabling proactive maintenance.

---

## 🏗 Architecture Overview

**Data Flow:**

1. Wio Terminal → MQTT Broker (JSON over MQTT)
2. MQTT Broker → Node-RED (JSON over MQTT)
3. Node-RED → Flask Backend (JSON over HTTP/HTTPS)
4. Backend → ML Models (feature vectors, internal)
5. Backend → Flutter App (JSON over HTTP/HTTPS)
6. Backend → FCM → Flutter (JSON over HTTPS)

---

## 🔧 IoT & Data Ingestion

* **Microcontroller:** Wio Terminal
* **Sensors:** Temperature, Humidity, Smoke, Vibration, Oil Level, Pressure
* **Messaging:** MQTT for lightweight, real-time telemetry
* **Node-RED:** Subscribes to MQTT, logs data (CSV), exposes latest snapshot via HTTP

---

## 🧠 Machine Learning

* **Models:** Random Forest classifiers
* **Purpose:** Predict component health (Good / Warning / Critical)
* **Training Data:** Sensor data collected via Node-RED
* **Why Random Forest:** Works well with small tabular datasets and noisy IoT data

---

## ⚙ Backend

* **Framework:** Flask
* **Responsibilities:**

  * Fetch live data from Node-RED
  * Preprocess sensor readings
  * Run ML predictions
  * Expose REST APIs for the mobile app
  * Trigger alerts (extensible)

---

## 📱 Frontend

* **Framework:** Flutter
* **Features:**

  * Live sensor visualization
  * Health status display
  * On-demand prediction triggering

---

## 📦 Tech Stack

### Backend

* Flask
* requests
* pandas, numpy
* scikit-learn
* csv, datetime

### Frontend

* Flutter (Material)
* GetX
* http
* syncfusion_flutter_gauges
* fl_chart

### IoT & Integration

* MQTT
* Node-RED

---

## 🗄 Storage

* **Current:** CSV-based logging (prototype)
* **Planned:** InfluxDB / MongoDB for scalable cloud storage

---

## 🔐 Security (Planned / Partial)

* HTTPS for APIs
* MQTT over TLS (MQTTS)
* JWT for authentication
* OAuth for service-to-service access

---

## 🚀 Future Enhancements

* Continuous background predictions
* Automated alerts (FCM / Email)
* Cloud database integration
* Fleet-level dashboards
* Role-based access control

---

## 📌 Summary

EcoSense demonstrates a complete **IoT → ML → Mobile** pipeline for predictive maintenance, designed to scale from a prototype to a production-ready system.

---


