# YouTube Video Generator

This project uses Hugging Face models to generate YouTube videos from text prompts.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Hugging Face token:
```
HUGGINGFACE_TOKEN=your_token_here
```

You can get your Hugging Face token from: https://huggingface.co/settings/tokens

## Usage

### Running the Gradio Client App

1. Ensure your virtual environment is activated.

2. Install the updated dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Hugging Face token as an environment variable named `HF_TOKEN`. In your terminal, you can do this temporarily:
```bash
export HF_TOKEN=your_token_here # On Windows: $env:HF_TOKEN="your_token_here"
```
For Google Colab, you can use the Secrets feature or set it in a code cell:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```
Replace `your_token_here` with your actual token.

4. Run the Gradio application:
```bash
python app.py
```

5. Gradio will provide a public URL (e.g., a Colab Share link) to access the image generator UI.

### Running the Script (The direct script `video_generator.py` is no longer the primary method for image generation with the UI.)

Run the video generator:
```bash
python video_generator.py
```

## Features

- Text-to-video generation using Hugging Face models
- Video enhancement capabilities
- Easy-to-use API for generating YouTube-ready videos

## Requirements

- Python 3.8+
- Hugging Face account and API token
- Sufficient disk space for video generation

## Note

This project uses the Hugging Face Inference API, which means you don't need to download the models locally. All processing is done through API calls. 