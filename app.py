import gradio as gr
from chatbot import chatbot

def main():
    """Create and launch the Gradio interface."""
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.Textbox(label="Ask your question:", lines=2, placeholder="Type your query here..."),
        outputs=gr.Textbox(label="Response"),
        title="SQL-Powered Chatbot",
        description="Ask questions, and the chatbot will query the database for you."
    )

    # Launch the interface
    iface.launch(share=False)

if __name__ == "__main__":
    main()
