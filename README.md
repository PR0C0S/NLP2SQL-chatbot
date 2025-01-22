# NLP to SQL Chatbot

This project is a NLP to SQL Chatbot designed to process user queries and fetch relevant information from a connected SQL database. The chatbot leverages OpenAI's GPT-based models for natural language understanding and Gradio for an intuitive user interface.

## Demo
Try the live demo: [E-commerce Chatbot Demo](https://huggingface.co/spaces/procos/ecommerce-v2)

## Features
- **Natural Language Querying**: Users can ask questions in plain English.
- **SQL Integration**: The chatbot queries the database to provide accurate responses.
- **User-Friendly Interface**: Built using Gradio for a seamless experience.
- **Customizable Models**: Powered by OpenAI's `gpt-4o-mini` with adjustable parameters.

## Architecture
For a detailed overview of the project architecture, check the [Architecture Documentation](docs/architecture.md).

## Test Methodology
For a detailed test methodologies of the project, check the [Test Documentation](docs/test_method.md).

## Installation

### Prerequisites
- Python 3.10
- An OpenAI API key
- Required Python dependencies (see `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/PR0C0S/NLP2SQL-chatbot.git
   cd ecommerce-chatbot

2. Install necessary requirements:
      ```bash
     pip install -r requirements.txt
     ```
3. Run the Application:
    ```bash
     python app.py
     ```