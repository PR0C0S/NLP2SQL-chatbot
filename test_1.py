import json
import pytest
from chatbot_test import chatbot

# Load the test dataset from a JSON file
@pytest.fixture
def test_dataset():
    test_dataset_path = "test1.json"  # Replace with the actual path to your JSON file
    with open(test_dataset_path, 'r') as file:
        return json.load(file)

# Function to test chatbot accuracy
def test_chatbot_accuracy(test_dataset):
    results = []
    for test_case in test_dataset:
        question = test_case["question"]
        expected_query = test_case["query"]

        # Generate the query using the chatbot
        response = chatbot(question)
        response_dict = json.loads(response)  # Convert the string to a dictionary
        generated_query = response_dict.get("captured_queries", "")

        # Check if the generated query matches the expected query
        is_correct = generated_query == expected_query
        results.append(is_correct)

    # Calculate accuracy
    total_cases = len(results)
    correct_cases = sum(results)
    accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0

    # Define an acceptable accuracy threshold
    acceptable_accuracy = 60.0  # Adjust based on your requirements

    print(f"Accuracy: {accuracy:.2f}% ({correct_cases}/{total_cases} correct cases)")

    # Assert that accuracy meets the acceptable threshold
    assert accuracy >= acceptable_accuracy, (
        f"Accuracy {accuracy:.2f}% is below the acceptable threshold of {acceptable_accuracy}%."
    )
