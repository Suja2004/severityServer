import numpy as np
import joblib

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

# Function to prompt the user for each response and display it
def get_responses():
    responses = []
    for question in question_labels:
        while True:
            try:
                response = int(input(f"{question} (0-10): "))
                if 0 <= response <= 4:
                    responses.append(response)
                    break
                else:
                    print("Please enter a number between 0 and 10.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    return responses

# Display the output format
def display_output(responses):
    print("\nPlease answer the following questions about your mental health. Enter a number between 0 and 10 where applicable.")
    for i in range(len(question_labels)):
        print(f"{question_labels[i]}: {responses[i]}")

# Function to make predictions using the trained model
def predict_disorder(responses):
    try:
        # Load saved artifacts
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_mappings = joblib.load('label_encoder.pkl')

        # Reverse the label encoder mappings using classes_
        reverse_label_mappings = {index: label for index, label in enumerate(label_mappings.classes_)}

        # Add the 'Total Score' feature by summing the responses
        total_score = sum(responses)
        responses_with_total_score = responses + [total_score]

        # Convert user responses to a NumPy array and preprocess it
        user_input = np.array(responses_with_total_score).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)
        severity = reverse_label_mappings[prediction[0]]

        print(f"\nBased on your responses, the PTSD severity is: {severity}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def main():
    print("Please answer the following questions about your mental health.")
    responses = get_responses()
    display_output(responses)
    predict_disorder(responses)

if __name__ == "__main__":
    main()
