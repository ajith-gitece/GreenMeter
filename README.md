# 🌿 GreenMeter - Interactive Energy & CO₂ Calculator
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("⚡ Welcome to GreenMeter – Smart Energy & Carbon Tracker 🌍\n")

# --- User Inputs ---
appliance = input("Enter appliance name: ")
hours = float(input("Enter hours used per day: "))
power = float(input("Enter power rating in watts (e.g. 1000 for 1kW device): "))

# --- Calculations ---
energy = (hours * power) / 1000  # kWh
co2 = energy * 0.82  # India factor

print(f"\n🔋 Appliance: {appliance}")
print(f"⏱️  Hours used: {hours} hr/day")
print(f"⚙️  Power rating: {power} W")
print(f"⚡ Energy consumed: {energy:.2f} kWh/day")
print(f"🌫️ CO₂ emitted: {co2:.2f} kg/day")

# --- Generate Past Data for Trend (Simulation) ---
days = list(range(1, 11))
usage = np.random.uniform(energy*0.7, energy*1.3, 10)  # random variation around today's usage
df = pd.DataFrame({"Day": days, "Energy_kWh": usage})

# --- Train a simple ML Model to Predict Next Day ---
X = np.array(df["Day"]).reshape(-1,1)
y = df["Energy_kWh"]
model = LinearRegression()
model.fit(X, y)
next_day_pred = model.predict([[11]])[0]
predicted_co2 = next_day_pred * 0.82

# --- Display Prediction ---
print(f"\n📈 Predicted energy for next day: {next_day_pred:.2f} kWh")
print(f"🌍 Predicted CO₂ emission: {predicted_co2:.2f} kg")

# --- Tip Generator ---
if co2 > 5:
    print("⚠️ Tip: Try reducing usage or switch to energy-efficient model to save up to 1–2 kg CO₂/day.")
elif co2 > 2:
    print("💡 Tip: Use smart timers or unplug idle devices to cut wastage.")
else:
    print("✅ Great! Your energy usage is eco-friendly.")

# --- Visualize the Trend ---
plt.figure(figsize=(6,4))
plt.plot(df["Day"], df["Energy_kWh"], marker='o', label='Past 10 Days Usage')
plt.axhline(next_day_pred, color='r', linestyle='--', label='Predicted Next Day')
plt.xlabel("Day")
plt.ylabel("Energy (kWh)")
plt.title(f"Energy Usage Trend for {appliance}")
plt.legend()
plt.grid(True)
plt.show()
