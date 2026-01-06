import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense

# --- 1. LOAD & PREPROCESS DATA ---
print("Loading Data...")
df_train = pd.read_csv('hvac_train_normal.csv')
df_test = pd.read_csv('hvac_test_mixed.csv')

# Drop Timestamp (Not useful for learning physics)
train_data = df_train.drop(['Timestamp'], axis=1)
# For test data, separate features and the ground truth label
test_data = df_test.drop(['Timestamp', 'Ground_Truth_Label'], axis=1)
test_labels = df_test['Ground_Truth_Label']

# Scale Data (Crucial for Neural Networks)
# We fit the scaler ONLY on training data to avoid data leakage
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# --- 2. BUILD AUTOENCODER MODEL ---
# The goal: Input -> Compress -> Reconstruct -> Output
input_dim = X_train_scaled.shape[1] # 4 Features

input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation="relu")(input_layer) # Compress
bottleneck = Dense(4, activation="relu")(encoder)  # Bottleneck
decoder = Dense(8, activation="relu")(bottleneck)  # Expand
output_layer = Dense(input_dim, activation="sigmoid")(decoder) # Reconstruct

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# --- 3. TRAIN MODEL ---
print("Training Autoencoder...")
history = autoencoder.fit(
    X_train_scaled, X_train_scaled, # Input and Target are the same (Unsupervised)
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    verbose=0
)

# --- 4. DETECT ANOMALIES ---
# Calculate Reconstruction Error on Test Data
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

# Define Threshold (Key Step)
# We set threshold at 99th percentile of the TRAINING errors (safe margin)
train_reconstructions = autoencoder.predict(X_train_scaled)
train_mse = np.mean(np.power(X_train_scaled - train_reconstructions, 2), axis=1)
threshold = np.quantile(train_mse, 0.99)

print(f"Anomaly Threshold calculated: {threshold:.5f}")

# Classify: If Error > Threshold, it's an Anomaly (1)
pred_labels = [1 if e > threshold else 0 for e in mse]

# --- 5. PRINT METRICS ---
print("\n--- RESULTS ---")
print(classification_report(test_labels, pred_labels, target_names=['Normal', 'Anomaly']))


# --- 6. Plot METRICS ---
# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- PLOT 1: MODEL CONVERGENCE (Training Loss) ---
# Discussion: "The model successfully learned the physics of the HVAC unit."
axes[0,0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0,0].plot(history.history['val_loss'], label='Val Loss', color='orange')
axes[0,0].set_title('Model Training Performance (Loss)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Epochs')
axes[0,0].set_ylabel('Reconstruction Error (MSE)')
axes[0,0].legend()

# --- PLOT 2: ANOMALY DETECTION OVER TIME (The "Money Shot") ---
# Discussion: "We can clearly see spikes in error when the fan and pressure fail."
axes[0,1].plot(mse, label='Reconstruction Error', color='red', alpha=0.7)
axes[0,1].axhline(y=threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
# Highlight failure regions
axes[0,1].axvspan(1500, 2500, color='yellow', alpha=0.3, label='Fan Failure Zone')
axes[0,1].axvspan(3500, 4500, color='orange', alpha=0.3, label='Pressure Leak Zone')
axes[0,1].set_title('Real-time Anomaly Detection', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Time Steps')
axes[0,1].set_ylabel('Error Score')
axes[0,1].legend(loc='upper right')

# --- PLOT 3: ERROR DISTRIBUTION (Histogram) ---
# Discussion: "The model cleanly separates Normal data (left) from Failures (right)."
normal_error = mse[test_labels == 0]
anomaly_error = mse[test_labels == 1]
axes[1,0].hist(normal_error, bins=50, alpha=0.7, color='blue', label='Normal Operation')
axes[1,0].hist(anomaly_error, bins=50, alpha=0.7, color='red', label='HVAC Failure')
axes[1,0].axvline(threshold, color='green', linestyle='--', label='Threshold')
axes[1,0].set_title('Separation of Normal vs Anomaly', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Reconstruction Error')
axes[1,0].set_yscale('log') # Log scale helps see small outlier counts
axes[1,0].legend()

# --- PLOT 4: CONFUSION MATRIX ---
# Discussion: "The system achieved high accuracy with minimal false alarms."
cm = confusion_matrix(test_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
axes[1,1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Actual Condition')
axes[1,1].set_xlabel('Predicted Condition')

plt.tight_layout()
plt.show()

# --- 6. ROOT CAUSE ANALYSIS (Which sensor triggered the alarm?) ---

# Calculate error for each sensor individually (do not average them yet)
# features: ['Temperature', 'Humidity', 'Pressure', 'Rotation']
feature_errors = np.power(X_test_scaled - reconstructions, 2)

# Create a DataFrame for easy plotting
df_errors = pd.DataFrame(feature_errors, columns=['Temperature', 'Humidity', 'Pressure', 'Rotation'])

# Plotting the "fingerprint" of the failures
plt.figure(figsize=(15, 6))

# Plot Temperature Error
plt.plot(df_errors['Temperature'], label='Temperature Error', color='red', alpha=0.8, linewidth=1.5)

# Plot Humidity Error
plt.plot(df_errors['Humidity'], label='Humidity Error', color='blue', alpha=0.6, linewidth=1.5)

# Add the failure zones for context
plt.axvspan(1500, 2500, color='yellow', alpha=0.2, label='Actual Fan Failure Zone')
plt.axvspan(3500, 4500, color='cyan', alpha=0.2, label='Actual Pressure Leak Zone')

plt.title('Root Cause Analysis: Temperature vs. Humidity Anomalies', fontsize=16, fontweight='bold')
plt.xlabel('Time Steps')
plt.ylabel('Reconstruction Error (Severity)')
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

# --- 7. SPLIT ROOT CAUSE ANALYSIS (The "Dashboard" View) ---

# Create a figure with 4 subplots stacked vertically
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# List of sensors and colors for distinction
sensors = ['Temperature', 'Humidity', 'Pressure', 'Rotation']
colors = ['red', 'blue', 'green', 'purple']

# Loop through each sensor to create its own plot
for i, sensor in enumerate(sensors):
    # Plot the reconstruction error for this specific sensor
    axes[i].plot(df_errors[sensor], color=colors[i], label=f'{sensor} Error', linewidth=1.2)

    # Highlight the Failure Zones on EVERY plot for easy comparison
    axes[i].axvspan(1500, 2500, color='yellow', alpha=0.3, label='Fan Failure Zone' if i == 0 else "")
    axes[i].axvspan(3500, 4500, color='cyan', alpha=0.3, label='Pressure Leak Zone' if i == 0 else "")

    # Formatting
    axes[i].set_ylabel('Error Score', fontweight='bold')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.5)

    # Add a title only to the top one
    if i == 0:
        axes[i].set_title('Root Cause Analysis: Sensor-wise Anomaly Detection', fontsize=16, fontweight='bold')

# Set the X-label only on the bottom plot
axes[3].set_xlabel('Time Steps', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()