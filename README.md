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

### Running the Flask App

1. Ensure your virtual environment is activated.

2. Install the updated dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and go to `http://127.0.0.1:5000/` to access the text generator UI.

### Running the Script (for direct text generation)

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