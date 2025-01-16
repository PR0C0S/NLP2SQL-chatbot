import os
from database import get_database
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

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