from diffusers import DiffusionPipeline
import gradio as gr
import torch
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model IDs
BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_MODEL_ID = "prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0"
TRIGGER_WORD = "Ultra realistic"

# Initialize the text-to-image pipeline globally
try:
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16)

    logger.info(f"Loading LoRA weights: {LORA_MODEL_ID}")
    pipe.load_lora_weights(LORA_MODEL_ID)

    # Move pipeline to GPU if available
    if torch.cuda.is_available():
        pipe.to("cuda")
        logger.info("Pipeline moved to GPU.")
    else:
        logger.warning("CUDA not available. Running on CPU, which may be slow.")

    image_generator = pipe
    logger.info("Text-to-image pipeline initialized.")

except Exception as e:
    logger.error(f"Error initializing text-to-image pipeline: {str(e)}")
    image_generator = None # Set to None if initialization fails

def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1 # Use -1 for random seed
):
    """
    Generate an image from a prompt using the Diffusion pipeline with LoRA weights.
    """
    if not image_generator:
        return None, "Error: Image generation pipeline not initialized. Check logs for errors."

    # Ensure trigger word is in the prompt
    if TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}"

    # Set seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 and torch.cuda.is_available() else None
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
        if torch.cuda.is_available():
             generator = torch.Generator("cuda").manual_seed(seed)
        else:
             generator = torch.Generator().manual_seed(seed)

    try:
        logger.info(f"Generating image for prompt: {prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Dimensions: {width}x{height}")
        logger.info(f"Steps: {num_inference_steps}")
        logger.info(f"Guidance scale: {guidance_scale}")
        logger.info(f"Seed: {seed}")

        # Generate the image
        output = image_generator(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        image = output.images[0]
        logger.info("Image generation successful.")
        return image, None

    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return None, f"Error generating image: {str(e)}"

# Create the Gradio interface
if image_generator:
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Prompt", lines=2, placeholder="Enter your positive prompt here..."),
            gr.Textbox(label="Negative Prompt", lines=2, placeholder="Enter your negative prompt here..."),
            gr.Number(label="Width", value=1024, precision=0),
            gr.Number(label="Height", value=1024, precision=0),
            gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=30, step=1),
            gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.1),
            gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Status/Error")
        ],
        title="FLUX Text-to-Image Generator with LoRA",
        description="Generate images using the FLUX model with UltraRealism LoRA."
    )

    if __name__ == "__main__":
        # Launch the Gradio interface
        interface.launch(share=True)
else:
    print("Image generation pipeline failed to initialize. Gradio interface will not be launched.") 