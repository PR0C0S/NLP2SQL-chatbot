import json
import os
from langchain.chains.sql_database.query import create_sql_query_chain
from database import get_database
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.chains.base import Chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
api_key=os.environ.get('OPENAI_API_KEY')

class EarlyStoppingException(Exception):
    """Custom exception to stop the chain early."""
    pass

template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Use the following format:

            Question: "Question here"
            SQLQuery: "SQL Query to run"
            SQLResult: "Result of the SQLQuery"
            Answer: Final answer here

            Only use the following tables:

            {table_info}.

            Question: {input}
            Thought:{agent_scratchpad}'''

# template = '''Answer the following questions as best you can. You have access to the following tools:

#         {tools}

#         Use the following format:

#         Question: the input question you must answer
#         Thought: you should always think about what to do
#         Action: the action to take, should be one of [{tool_names}]
#         Action Input: the input to the action
#         Observation: the result of the action
#         ... (this Thought/Action/Action Input/Observation can repeat N times)
#         Thought: I now know the final answer
#         Final Answer: the final answer to the original input question

#         Begin!

#         Question: {input}
#         Thought:{agent_scratchpad}'''
# sql_prompt_template = PromptTemplate(
#     input_variables=['dialect', 'table_info', 'input', 'top_k','agent_scratchpad'],
#     template=template
# )
prompt = PromptTemplate.from_template(template)


class QueryCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.queries = ""
        self.stop_execution  = False
        

    def on_agent_action(self, action, *, run_id, parent_run_id = None, **kwargs):
        if action.tool == "sql_db_query_checker":
            # print(f"Tool Invoked: {action.tool}, Input: {action.tool_input}")  # Debugging
            self.queries = action.tool_input['query']
            self.stop_execution  = True
            raise EarlyStoppingException

def chatbot(query_text):
    """Handles chatbot queries."""
    try:
        db = get_database()
        if not isinstance(db, SQLDatabase):
            raise ValueError("Database connection must be of type SQLDatabase.")
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key,temperature=0)
        callback_handler = QueryCaptureCallback()
        sql_chain= create_sql_agent(
            llm, 
            db=db, 
            agent_type="openai-tools", 
            verbose=False, 
            top_k=5,
            agent_executor_kwargs={"callbacks": [callback_handler]},
        )
        response = sql_chain.invoke({"input": query_text})
       
        captured_queries = callback_handler.queries

        cleaned_query = captured_queries.replace('\n', ' ').strip()
        
        response={
            "output": response["output"],  # Agent's response
            "captured_queries": cleaned_query,
        }
        return json.dumps(response)
    except EarlyStoppingException:
        # Handle early stopping and return the captured query
        captured_query = callback_handler.captured_query or "No query captured"
        return json.dumps({
            "output": "",
            "captured_queries": captured_query
        })    
    except Exception as e:
        captured_queries = callback_handler.queries
        cleaned_query = captured_queries.replace('\n', ' ').strip()
        response={
            "output": "",  # Agent's response
            "captured_queries": cleaned_query,
        }
        return json.dumps(response)

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