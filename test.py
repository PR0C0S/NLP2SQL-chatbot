from difflib import SequenceMatcher
import os
from gradio_client import Client
import json
from chatbot_test import chatbot
import time

embedding_key=os.environ.get('OPENAI_EMBEDDING_KEY')
api_key=os.environ.get('OPENAI_API_KEY')


def load_test_dataset(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)


def is_similar_query(generated_query, expected_query,threshold):
    """
    Compare two SQL queries for similarity.

    Args:
        generated_query (str): The generated SQL query.
        expected_query (str): The expected SQL query.
        threshold (float): Similarity threshold (0.0 to 1.0).

    Returns:
        bool: True if similarity ratio >= threshold, else False.
    """
    # Normalize whitespace and case
    generated_query = " ".join(generated_query.strip().lower().split())
    if not generated_query.endswith(";"):
        generated_query += ";"
    expected_query = " ".join(expected_query.strip().lower().split())

    # Calculate similarity ratio
    similarity_ratio = SequenceMatcher(None, generated_query, expected_query).ratio()
    is_correct = similarity_ratio >= threshold  # Decide if it's correct
    return similarity_ratio, is_correct
    return similarity_ratio

def test_chatbot_with_gradio(test_dataset):
    start_time = time.time()
    results = []
    for test_case in test_dataset:
        try: 
            question = test_case["question"]
            expected_query = test_case["query"]

            # Invoke the chatbot using the Gradio client
            # response = client.predict(
            #     query_text=question,
            #     api_name=api_name
            # )
            response=chatbot(question)

            response_dict = json.loads(response) 
            captured_queries = response_dict.get("captured_queries", "")
          
            # Compare with the expected query
            similarity_score, is_correct = is_similar_query(captured_queries, expected_query,threshold=0.8)

            results.append({
                "question": question,
                "expected_query": expected_query,
                "generated_query": captured_queries,
                "similarity_score": similarity_score,
                "is_correct": is_correct
            })
        except Exception as e:
            print(f"ERROR:{e}")
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        
    return results

# Function to print the test results
def print_test_results(results):
    for i, result in enumerate(results, start=1):
        print(f"Test Case {i}:")
        print(f"  Question: {result['question']}")
        print(f"  Expected Query: {result['expected_query']}")
        print(f"  Generated Query: {result['generated_query']}")
        print(f"  Similarity Score: {result['similarity_score']:.2f}")
        print(f"  Is Correct: {'Yes' if result['is_correct'] else 'No'}")
        print("-" * 60)

def calculate_accuracy(results):
    total_cases = len(results)
    correct_cases = sum(result["is_correct"] for result in results)
    total_similarity_score = sum(result["similarity_score"] for result in results)
    accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0
    average_similarity = (total_similarity_score / total_cases) if total_cases > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_cases}/{total_cases} correct cases)")
    print(f"Average Similarity Score: {average_similarity:.2f}")
    return accuracy

# Main execution
if __name__ == "__main__":
    # Load the test dataset from a JSON file
    test_dataset_path = "test_case.json"  # Replace with the actual path to your JSON file
    test_dataset = load_test_dataset(test_dataset_path)

    # # Initialize the Gradio client

    # Test the chatbot with the dataset
    test_results = test_chatbot_with_gradio(  # Replace with your Gradio app's API name
        test_dataset=test_dataset
    )

    # Print the results
    print_test_results(test_results)
    calculate_accuracy(test_results)
