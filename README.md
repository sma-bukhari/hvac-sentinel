# HVAC Sentinel: Digital Twin for Predictive Maintenance
**A Low-Cost AI Framework for Carbon Reduction in Public Infrastructure**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-ESP8266%20%7C%20Arduino-orange)

## 📌 Overview
HVAC Sentinel is a research initiative and digital tool designed to democratize predictive maintenance for the Global South. By combining **Digital Twin methodology** with low-cost IoT hardware (ESP8266), this project detects HVAC anomalies—such as refrigerant leaks or fan degradation—before catastrophic failure.

This repository contains the simulation code, firmware logic, and technical documentation for the proposed framework.

## 🚀 Key Features
* **Deep Learning Autoencoder:** Unsupervised learning model for anomaly detection.
* **Low-Cost Architecture:** Designed for retrofit on legacy equipment (<$30 hardware cost).
* **Sustainability Focus:** Directly targets **SDG 13 (Climate Action)** by optimizing energy efficiency.

## 📊 Technical Approach
The system monitors four key variables:
1.  **Temperature & Humidity** (DHT11)
2.  **Pressure** (BMP180)
3.  **Rotation/RPM** (IR Sensor)

Data is processed via an Autoencoder network that learns the statistical distribution of "normal" operations. High reconstruction errors trigger real-time alerts.


## 📂 Repository Structure
* `/scripts` - Python scripts for synthetic data generation and model training.
* `/firmware` - Arduino/C++ code for the ESP8266 sensor node.
* `/docs` - Technical Whitepaper and project proposal.

## 🔗 Contact & Citation
**Syed Muhammad Ali Bukhari**
*National University of Sciences and Technology (NUST)*
