from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self):
        # Initialize the text generation pipeline
        self.pipe = pipeline("text-generation", model="openai-community/gpt2-large")
        logger.info("Text generation pipeline initialized.")

    def generate_text(self, prompt, max_new_tokens=100):
        """
        Generate text from a prompt using the transformers pipeline
        """
        try:
            logger.info(f"Generating text from prompt: {prompt}")

            # Perform text generation using the pipeline
            # The pipeline returns a list of dictionaries, we want the 'generated_text' from the first item
            output = self.pipe(prompt, max_new_tokens=max_new_tokens)
            generated_text = output[0]['generated_text']

            logger.info("Text generation successful.")
            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

def main():
    # Example usage
    generator = TextGenerator()
    
    # Generate text
    prompt = "Once upon a time," # Example prompt
    generated_text = generator.generate_text(prompt)
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("---------------------\n")

if __name__ == "__main__":
    main() 