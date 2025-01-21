import json
from langchain_openai import ChatOpenAI
import openai
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import time
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.callbacks.base import BaseCallbackHandler

api_key=os.environ.get('OPENAI_API_KEY')
# Configuration
host = os.environ.get('HOST')
user = os.environ.get('USER')
password = os.environ.get('PASSWORD')
database = os.environ.get('DATABASE')
DATABASE_URI = f'mysql+pymysql://{user}:{password}@{host}/{database}'

# Initialize SentenceTransformer and Chroma
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
CHROMA_PATH = "chroma"  # Specify the directory for Chroma persistence

# Use PersistentClient for the updated Chroma configuration
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection("test_results", embedding_function=embedding_function)

def connect_to_db():
    try:
        engine = create_engine(DATABASE_URI, echo=True)
        Session = sessionmaker(bind=engine)
        session = Session()
        print("Database connection successful.")
        return session, engine
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None, None

def store_in_chroma(test_name, result):
    # Convert the result to a string and create an embedding
    result_str = json.dumps(result, default=str)
    embeddings = embedding_function([result_str])
    
    # Generate a unique ID using test_name and timestamp
    unique_id = f"{test_name}_{int(time.time())}"
    
    # Store the result in Chroma with the unique ID
    chroma_collection.add(
        embeddings=embeddings,
        metadatas=[{"test_name": test_name}],
        ids=[unique_id]
    )
    print(f"Stored in Chroma with ID: {unique_id}")

def calculate_similarity(expected_result, generated_result):
    # Convert both expected and generated result to string and create embeddings
    expected_str = json.dumps(expected_result, default=str)
    generated_str = json.dumps(generated_result, default=str)
    
    embeddings = embedding_function([expected_str, generated_str])
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

class QueryCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.captured_query = ""  # Properly define the attribute
        self.stop_execution = False

    def on_agent_action(self, action, *, run_id, parent_run_id=None, **kwargs):
        if action.tool == "sql_db_query_checker":
            self.captured_query = action.tool_input.get("query", "")  # Capture the query
            self.stop_execution = True
            raise EarlyStoppingException


class EarlyStoppingException(Exception):
    """Custom exception to stop the chain early."""
    pass


def run_tests(json_file, session, engine):
    similarity_scores = []  # List to store similarity scores for each test
    start_time = time.time()

    with open(json_file, 'r') as f:
        test_cases = json.load(f)
    
    iter_count=1

    for test in test_cases:
        question = test['question']
        expected_answer = test['expected_answer']
        print(f"\nQuestion {iter_count}: {question}")
        print(f"Expected SQL: {expected_answer}")
        try:
            # Execute the expected SQL query and fetch the result
            with engine.connect() as conn:
                result = conn.execute(text(expected_answer)).fetchall()
            print(f"Result from DB: {result}")

                # Store the result in Chroma (you can customize this storage if necessary)
            store_in_chroma(test['question'], result)

                # Generate SQL using the chatbot_test function and capture the raw result (including query)
            generated_response = chatbot_test(f"Question: {question}")

                # Extract the SQL query from the generated response
            generated_sql = json.loads(generated_response).get("captured_query")
            print(f"Generated SQL by chatbot_test: {generated_sql}")

            if generated_sql:
                    # Execute the generated SQL query and get the result
                    generated_result = conn.execute(text(generated_sql)).fetchall()
                    print(f"Generated Result: {generated_result}")

                    # Calculate similarity between expected and generated results
                    similarity = calculate_similarity(result, generated_result)
                    print(f"Similarity Score: {similarity:.2f}")

                    # Store the similarity score for calculating the grand global score
                    similarity_scores.append(similarity)
            else:
                    print("❌ No valid SQL query generated.")

        except Exception as e:
            print(f"❌ Test failed. Query execution error: {e}")
        
        end_one_time = time.time()
        time_taken = end_one_time - start_time
        print(f"\nTotal Time Taken for test {iter_count}: {time_taken:.2f} seconds")
        iter_count+=1

    # Calculate and print the grand global score (average similarity score)
    if similarity_scores:
        grand_global_score = sum(similarity_scores) / len(similarity_scores)
        print(f"\nGrand Global Score (Average Similarity): {grand_global_score:.2f}")
    else:
        print("\nNo similarity scores to calculate.")
    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"\nTotal Time Taken for run_tests: {total_time_taken:.2f} seconds")

def chatbot_test(query_text):
    """Handles chatbot queries and returns the generated SQL query."""
    print("ENTERED CHATBOT TEST////////////")
    try:
        # Step 1: Initialize database and LLM
        db = SQLDatabase.from_uri(
        DATABASE_URI,
        ignore_tables=['ecommerce_smaller']
    )
        if not isinstance(db, SQLDatabase):
            raise ValueError("Database connection must be of type SQLDatabase.")
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        
        # Initialize the callback handler to capture the query
        callback_handler = QueryCaptureCallback()

        # Step 2: Create the SQL agent with the callback
        sql_chain = create_sql_agent(
            llm,
            db=db,
            agent_type="openai-tools",
            verbose=False,
            top_k=5,
            agent_executor_kwargs={"callbacks": [callback_handler]},
        )

        # Step 3: Invoke the chain
        response = sql_chain.invoke({"input": query_text})

        # Step 4: Get the captured query
        captured_query = callback_handler.captured_query or "No query captured"
        
        # Clean up the query
        cleaned_query = captured_query.replace('\n', ' ').strip()

        return json.dumps({
            "captured_query": cleaned_query,
            "output": response.get("output", "")
        })
    except OperationalError as e:
        # Handle database connection errors
        print(f"❌ Database connection failed: {e}")
        return json.dumps({
            "error": f"Database connection error: {str(e)}"
        })    
    except EarlyStoppingException:
        # Handle early stopping and return the captured query
        captured_query = callback_handler.captured_query or "No query captured"
        return json.dumps({
            "captured_query": captured_query,
            "output": ""
        })
    except openai.APIConnectionError :
        print("❌ OPENAI connection error")
        return json.dumps({
            "error": "An error occurred"
        })   
    except Exception as e:
        print(f"Error:{e}")
        return json.dumps({
            "error": f"An error occurred: {str(e)}"
        })


def main():
    # Connect to the database
    session, engine = connect_to_db()
    if not session or not engine:
        print("Skipping tests due to database connection failure.")
        return

    # Run tests from JSON file
    json_file = "tests/test_cases.json"
    run_tests(json_file, session, engine)

if __name__ == "__main__":
    main()
