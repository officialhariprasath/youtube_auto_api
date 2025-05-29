from transformers import pipeline
import gradio as gr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the text generation pipeline globally
try:
    text_generator = pipeline("text-generation", model="openai-community/gpt2-large")
    logger.info("Text generation pipeline initialized.")
except Exception as e:
    logger.error(f"Error initializing text generation pipeline: {str(e)}")
    text_generator = None # Set to None if initialization fails

def generate_text(prompt):
    """
    Generate text from a prompt using the transformers pipeline
    """
    if not text_generator:
        return "Error: Text generation pipeline not initialized. Check logs for errors."

    try:
        logger.info(f"Generating text for prompt: {prompt}")
        # The pipeline returns a list of dictionaries, we want the 'generated_text' from the first item
        output = text_generator(prompt, max_new_tokens=100)
        generated_text = output[0]['generated_text']
        logger.info("Text generation successful.")
        return generated_text

    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return f"Error generating text: {str(e)}"

# Create the Gradio interface
if text_generator:
    interface = gr.Interface(
        fn=generate_text,
        inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        outputs="text",
        title="Hugging Face Text Generator (GPT-2 Large)",
        description="Enter a prompt and the model will generate text."
    )

    if __name__ == "__main__":
        # Launch the Gradio interface
        # In Google Colab, this will provide a public URL
        interface.launch(share=True)
else:
    print("Text generation pipeline failed to initialize. Gradio interface will not be launched.") 