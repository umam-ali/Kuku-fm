# AI Powered Thumbnail Image Generator

This Streamlit-based application generates artistic image concepts by combining creative prompts with advanced image generation and post-processing techniques. It leverages GPT-4 for generating stylistic prompts and Playground 2.5 aesthetic for image synthesis. The final output includes images enhanced with gradient fades, dynamic title overlays, logos, and aesthetic quality scoring.

## Features

- **Artistic Prompt Generation**  
  - Uses GPT-4 to generate 10 distinct artistic prompts based on a user-provided title and concept note.
  - Each prompt adheres to specific guidelines, including a mixture of visual styles (portrait, scenic, abstract) and unique scene descriptions.
  - Prompts include key metadata such as artistic style, sub-style, font selection, font color, and glow color.

- **Image Generation with Playground 2.5 aesthetic**  
  - Integrates Playground 2.5 aesthetic to generate high-quality images from textual prompts.
  - Customizable generation parameters including image dimensions, inference steps, guidance scale, and seed for reproducibility.

- **Post-Processing & Enhancement**  
  - **Fading Effects:** Applies fade effects to different parts of the image (top, middle, and bottom) to achieve a layered visual style.
  - **Dynamic Title Overlays:** Uses selected fonts and dynamically adjusts font and glow colors for optimal contrast.
  - **Logo Integration:** Overlays a predefined transparent logo onto the image.
  - **Aesthetic Evaluation:** Utilizes a pre-trained MLP-based aesthetic predictor to evaluate and score generated images.
  - **Image Ranking:** Selects the final image based on aesthetic scores with a randomized selection among the top two scoring candidates.

## Key Components

1. **Prompt Generation:**  
   - Communicates with OpenAI's Chat API (GPT-4) to generate artistic prompts in JSON format.
   - Uses a pre-defined dictionary of artistic styles and fonts to ensure stylistically consistent outputs.

2. **Image Synthesis (`playground` function):**  
   - Loads the Playground 2.5 pipelineS from a specified model file.
   - Generates an image from the provided prompt using user-defined generation parameters.

3. **Aesthetic Scoring (`aesthetic` function):**  
   - Loads and evaluates images using a pre-trained and fine-tuned MLP model.
   - Utilizes CLIP features to create an embedding that is used by the MLP for scoring.

4. **Image Post-Processing:**  
   - **apply_fade:** Creates fading effects for specified regions of the image.
   - **adjust_font_colors_for_contrast:** Adjusts font and glow colors based on background brightness.
   - **draw_title:** Dynamically overlays a title on the image with optional glow effects.
   - **add_logo_to_image:** Overlays a transparent logo onto the image.

5. **User Interface & Flow (Streamlit):**  
   - Provides a web interface for users to input a title and concept note.
   - Displays generated prompts, images for each artistic style, and final ranked images based on aesthetic scoring.
   - Implements retries to ensure valid prompt generation.

## Setup Instructions

### Prerequisites

- Python 3.8 or newer.
- Playground 2.5 aesthetic from https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic
- Aesthetic CLP+MLP model from this repository i.e sac+logos+ava1-l14-linearMSE.pth
- Fonts from this repository
- Kuku Fm logo from this repository
- Required libraries:
  - `streamlit`
  - `openai`
  - `torch`
  - `clip`
  - `diffusers`
  - `numpy`
  - `opencv-python`
  - `Pillow`

Install the necessary packages via pip:

```bash
pip install streamlit openai torch clip diffusers numpy opencv-python Pillow
