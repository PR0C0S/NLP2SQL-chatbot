# Project Architecture for SQL-Powered Chatbot

## `app.py`
The `app.py` file serves as the entry point for the project. It creates and launches a Gradio interface that users can interact with to ask questions. The file contains the following key elements:

1. **Gradio Interface**:
   - A Gradio `Interface` is defined to connect the user input (a query) with the chatbot function.
   - `inputs`: A Gradio `Textbox` where users type their questions.
   - `outputs`: A Gradio `Textbox` where the chatbot's responses are displayed.
   - `title` and `description`: Provide context and guidance for users about the chatbot.

2. **Launching the Interface**:
   - The `iface.launch()` method starts the Gradio application locally, allowing users to interact with the chatbot via a web-based UI.

## `chatbot.py`
The `chatbot.py` file implements the core logic for processing user queries and interacting with the database. The key components are:

1. **Database Retrieval**:
   - The `get_database()` function (from `database.py`) retrieves the SQL database connection. It ensures the database is of type `SQLDatabase`.

2. **Language Model**:
   - `ChatOpenAI` is initialized with the `gpt-4o-mini` model, using the `OPENAI_API_KEY` for authentication.
   - The `temperature` parameter is set to `0` to ensure deterministic and focused responses.

3. **SQL Agent Creation**:
   - The `create_sql_agent` function creates an agent that can query the SQL database using the language model.
   - `agent_type="openai-tools"` specifies the type of agent, and `top_k=5` limits the query's top results.

4. **Query Handling**:
   - The `query_text` input is processed through the SQL agent (`sql_chain.invoke()`), generating a response that combines language understanding with database query results.

5. **Error Handling**:
   - Errors during processing are caught and returned as part of the response to ensure the application does not crash.

## Flow Diagram
1. **User Query**:
   - A user inputs a question in the Gradio interface.
2. **Processing**:
   - The `app.py` routes the query to `chatbot.py`, which processes it using a language model and the SQL database.
3. **Database Interaction**:
   - The SQL agent queries the database for relevant information based on the user's input.
4. **Response Generation**:
   - The chatbot generates a response using the results from the database and sends it back to the Gradio interface.
5. **User Response**:
   - The processed response is displayed to the user.

## Environment Variables
- `OPENAI_API_KEY`: Required for authenticating with the OpenAI API.

## Dependencies
- **Gradio**: For creating the user interface.
- **langchain_community**: For managing the language model and creating the SQL agent.
- **SQLDatabase**: For interacting with the database.

## Conclusion
This project combines natural language processing with SQL database querying to provide a conversational interface that can retrieve and present relevant information based on user queries. The modular structure ensures clear separation of responsibilities, making it easier to maintain and extend.
