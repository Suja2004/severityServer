from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

CORS(app)  

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_mappings = joblib.load('label_encoder.pkl')

# Reverse the label encoder mappings using classes_
reverse_label_mappings = {index: label for index, label in enumerate(label_mappings.classes_)}

# Define the questions (labels) for your mental health survey
question_labels = [
    "Unwanted Memories Frequency",
    "Upsetting Dreams Frequency",
    "Feeling Upset When Reminded",
    "Avoiding Thoughts or Feelings",
    "Avoiding External Reminders",
    "Emotional Numbness",
    "Being Overly Alert or On Edge",
    "Difficulty Concentrating",
    "Irritability or Anger Outbursts",
    "Physical Reactions When Reminded"
]

@app.route('/predict', methods=['POST'])
def predict_disorder():
    try:
        # Get responses from the frontend
        data = request.get_json()
        responses = data.get('responses', [])

        # Add the 'Total Score' feature by summing the responses
        total_score = sum(responses)
        responses_with_total_score = responses + [total_score]

        # Convert user responses to a NumPy array and preprocess it
        user_input = np.array(responses_with_total_score).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)
        severity = reverse_label_mappings[prediction[0]]

        return jsonify({'severity': severity})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, port=3000)
