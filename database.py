import os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

host=os.environ.get('HOST')
user=os.environ.get('USER')
password=os.environ.get('PASSWORD')
database=os.environ.get('DATABASE')

def get_database():
    """Connect to the database."""
    db = SQLDatabase.from_uri(
        f'mysql+pymysql://{user}:{password}@{host}/{database}',
        ignore_tables=['ecommerce_smaller']
    )
    return db

def execute_query(db):
    return QuerySQLDataBaseTool(db=db)
