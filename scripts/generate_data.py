import pandas as pd
import numpy as np

# --- SETTINGS ---
train_size = 10000
test_size = 5000

# --- HELPER: THERMODYNAMIC COUPLING ---
def get_target_humidity(temp, base_temp=24.0, base_hum=45.0):
    # Rule of thumb: For every 1°C rise, RH drops ~2-3% if moisture is constant.
    # We use a factor of 1.5 for stability in this simulation.
    return base_hum - (temp - base_temp) * 1.5

# --- 1. GENERATE TRAINING DATA (Normal Operation with Physics) ---
print("Generating Training Data (Normal Operation)...")

data_train = []
# Initialize base values
temp = 24.0
hum = 45.0
press = 1013.25
rpm = 1200

for t in range(train_size):
    # 1. Simulate environmental drift (e.g., Day/Night cycle)
    # Temp oscillates slowly
    temp_drift = np.sin(t / 500) * 0.5

    # 2. Apply Coupling to Base Values
    current_base_temp = 24.0 + temp_drift
    # Humidity opposes temperature naturally (Psychrometric link)
    current_base_hum = get_target_humidity(current_base_temp)

    # 3. Add Sensor Noise
    row = [
        t,
        round(current_base_temp + np.random.normal(0, 0.1), 2),
        round(current_base_hum + np.random.normal(0, 0.2), 2),
        round(1013.25 + np.random.normal(0, 0.05), 2),
        int(1200 + np.random.normal(0, 5))
    ]
    data_train.append(row)

df_train = pd.DataFrame(data_train, columns=['Timestamp', 'Temperature', 'Humidity', 'Pressure', 'Rotation'])
df_train.to_csv('hvac_train_normal.csv', index=False)


# --- 2. GENERATE TEST DATA (With Thermodynamically Accurate Failures) ---
print("Generating Test Data (With Failures)...")

data_test = []
t_offset = train_size

# State Variables
curr_temp = 24.0
curr_hum = 45.0
curr_press = 1013.25
curr_rpm = 1200

# Physics constants (Inertia factors)
cooling_capacity = 0.1 # How fast temp recovers
rpm_recovery = 0.1
press_recovery = 0.1

for t in range(test_size):
    ts = t + t_offset

    # --- A. CALCULATE PHYSICS TARGETS ---
    # Normal target humidity based on current temp
    target_hum = get_target_humidity(curr_temp)

    # --- B. INJECT FAULTS ---

    # ZONE 1: FAN FAILURE (Rows 1500-2500)
    # Cause: Motor electrical fault
    # Physics Chain:
    # 1. RPM drops (Primary)
    # 2. Cooling fails -> Temp rises (Secondary)
    # 3. Temp rises -> Humidity drops (Thermodynamic Law)
    # 4. Airflow stops -> Static Pressure drops slightly (Mechanical Law)
    if 1500 <= t < 2500:
        curr_rpm -= 0.8  # RPM decays
        curr_temp += 0.015 # Temp rises due to lack of airflow
        curr_press -= 0.01 # NEW: Slight pressure drop due to lack of "push"
        label = 1

        # Note: Humidity naturally drops toward target_hum in step C below

    # ZONE 2: PRESSURE LEAK (Rows 3500-4500)
    # Cause: Chamber door open or seal break
    # Physics Chain:
    # 1. Pressure drops significantly (Primary)
    # 2. External moist air enters -> Humidity rises (Secondary)
    # 3. Fan load changes -> RPM jitter (Mechanical Side Effect)
    elif 3500 <= t < 4500:
        curr_press -= 0.05 # Significant Pressure loss
        label = 1

        # Physics Override: Moisture ingress!
        # Instead of following the temp relationship, humidity rises due to leak
        target_hum = 55.0 # External air is more humid

        # Physics Side Effect: Turbulence
        # Broken seal causes fan load fluctuation
        curr_rpm += np.random.normal(0, 3) # Extra jitter

    # ZONE 3: NORMAL OPERATION / RECOVERY
    else:
        # Recover variables to setpoint
        curr_rpm += (1200 - curr_rpm) * rpm_recovery
        curr_temp += (24.0 - curr_temp) * cooling_capacity
        curr_press += (1013.25 - curr_press) * press_recovery
        label = 0

    # --- C. UPDATE STATE (With Inertia) ---
    # Humidity moves towards its target (it doesn't snap instantly)
    curr_hum += (target_hum - curr_hum) * 0.1

    # Clamp Limits
    curr_rpm = max(0, curr_rpm)

    # --- D. ADD NOISE & SAVE ---
    data_test.append([
        ts,
        round(curr_temp + np.random.normal(0, 0.1), 2),
        round(curr_hum + np.random.normal(0, 0.2), 2),
        round(curr_press + np.random.normal(0, 0.05), 2),
        int(curr_rpm + np.random.normal(0, 5)),
        label
    ])

df_test = pd.DataFrame(data_test, columns=['Timestamp', 'Temperature', 'Humidity', 'Pressure', 'Rotation', 'Ground_Truth_Label'])
df_test.to_csv('hvac_test_mixed.csv', index=False)

print(f"Done. Files created with full thermodynamic coupling:\n1. hvac_train_normal.csv\n2. hvac_test_mixed.csv")