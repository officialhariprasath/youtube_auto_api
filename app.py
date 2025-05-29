from flask import Flask, render_template, request
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the text generation pipeline globally
# This will download the model the first time the app runs
try:
    text_generator = pipeline("text-generation", model="openai-community/gpt2-large")
    logger.info("Text generation pipeline initialized.")
except Exception as e:
    logger.error(f"Error initializing text generation pipeline: {str(e)}")
    text_generator = None # Set to None if initialization fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    generated_text = ""
    error_message = None

    if text_generator:
        try:
            logger.info(f"Generating text for prompt: {prompt}")
            # The pipeline returns a list of dictionaries, we want the 'generated_text' from the first item
            output = text_generator(prompt, max_new_tokens=100)
            generated_text = output[0]['generated_text']
            logger.info("Text generation successful.")
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            error_message = f"Error generating text: {str(e)}"
    else:
        error_message = "Text generation pipeline not initialized. Check logs for errors."
        logger.error(error_message)

    return render_template('index.html', prompt=prompt, generated_text=generated_text, error_message=error_message)

if __name__ == '__main__':
    # Note: In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    app.run(debug=True) 