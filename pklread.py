import joblib
import numpy as np
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Example data (new responses)
new_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,56]).reshape(1, -1)  # Replace with new responses
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)
severity = label_encoder.inverse_transform(prediction)
print(f"Predicted Severity: {severity[0]}")
