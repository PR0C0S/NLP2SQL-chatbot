import json
import os
from database import get_database
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.callbacks.base import BaseCallbackHandler

api_key=os.environ.get('OPENAI_API_KEY')

        
def chatbot(query_text):
    """Handles chatbot queries."""
    try:
        # Get Database
        db = get_database()
        if not isinstance(db, SQLDatabase):
            raise ValueError("Database connection must be of type SQLDatabase.")
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key,temperature=0)
        sql_chain= create_sql_agent(
            llm, 
            db=db, 
            agent_type="openai-tools", 
            verbose=True, 
            top_k=5,
        )
        response = sql_chain.invoke({"input": query_text})
       
        return response['output']
        
    except Exception as e:
  
        return f"Error:{e}"
    
class EarlyStoppingException(Exception):
    """Custom exception to stop the chain early."""
    pass


class QueryCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.captured_query = None
        self.stop_execution = False

    def on_agent_action(self, action, *, run_id, parent_run_id=None, **kwargs):
        if action.tool == "sql_db_query_checker":
            # Capture the query
            self.captured_query = action.tool_input['query']
            self.stop_execution = True
            raise EarlyStoppingException


def chatbot_test(query_text):
    """Handles chatbot queries and returns the generated SQL query."""
    try:
        # Step 1: Initialize database and LLM
        db = get_database()
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
        
    except EarlyStoppingException:
        # Handle early stopping and return the captured query
        captured_query = callback_handler.captured_query or "No query captured"
        return json.dumps({
            "captured_query": captured_query,
            "output": ""
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"An error occurred: {str(e)}"
        })