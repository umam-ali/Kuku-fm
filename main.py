import streamlit as st
import openai
import os
import json
from PIL import Image
import numpy as np
import cv2
from PIL import ImageDraw, ImageFont
import sys
import torch
import clip
from PIL import Image
import torch.nn as nn
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
import random

#function to generate images
def playground(prompt,path):
    model_path = r"\playground-v2.5-1024px-aesthetic.fp16.safetensors"
    output_dir = r"\Generated Images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generation parameters
    negative_prompt = "blurry, low quality, distorted, out of frame, cropped, extra limbs, ugly, deformed, bad anatomy, poor lighting, unnatural colors"
    image_width = 1024
    image_height = 1024
    num_inference_steps = 70
    guidance_scale = 7
    seed_value = 42
    
    # Load the SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    
    # Enable optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    
    # Generate image
    generator = torch.Generator(device="cuda").manual_seed(seed_value)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        # SDXL specific parameters
        target_size=(image_width, image_height),
        original_size=(image_width, image_height)
    )
    
    # Save image
    image_path = os.path.join(output_dir, path+".png")
    result.images[0].save(image_path)
    return image_path

def aesthetic(image_path):
    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )
    
        def forward(self, x):
            return self.layers(x)
    
    
    
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the aesthetic predictor model (MLP) with input dimension 768
    model = MLP(768)
    state_dict = torch.load(r"\sac+logos+ava1-l14-linearMSE.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load CLIP model and preprocessing
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    
    # Load and preprocess your image
    pil_image = Image.open(image_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        im_tensor = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        prediction = model(im_tensor)
    
    return prediction.item()


openai.api_key = "ENTER YOUR API KEY HERE"

available_fonts = {
    "Cinzel": "Cinzel-Regular.ttf",
    "PlayfairDisplay": "PlayfairDisplay-Regular.ttf",
    "Anton": "Anton-Regular.ttf",
    "BebasNeue": "BebasNeue-Regular.ttf",
    "Oswald": "Oswald-Regular.ttf",
    "Lobster": "Lobster-Regular.ttf",
    "Pacifico": "Pacifico-Regular.ttf",
    "Lora": "Lora-Regular.ttf",
    "Raleway": "Raleway-Regular.ttf",
    "AbrilFatface": "AbrilFatface-Regular.ttf",
    "AmaticSC": "AmaticSC-Regular.ttf",
    "Orbitron": "Orbitron-Regular.ttf",
    "Audiowide": "Audiowide-Regular.ttf",
    "Exo": "Exo-Regular.ttf",
    "ComicNeue": "ComicNeue-Regular.ttf",
    "VT323": "VT323-Regular.ttf",
    "PressStart2P": "PressStart2P-Regular.ttf",
    "DancingScript": "DancingScript-Regular.ttf",
    "IndieFlower": "IndieFlower-Regular.ttf",
    "JosefinSans": "JosefinSans-Regular.ttf",
    "UncialAntiqua": "UncialAntiqua-Regular.ttf",
    "MedievalSharp": "MedievalSharp-Regular.ttf",
    "Bangers": "Bangers-Regular.ttf",
    "FiraSans": "FiraSans-Regular.ttf"
}

art_styles = {
    "Anime/Manga": [
        "Shonen Action",
        "Shojo Romance",
        "Mecha",
        "Chibi",
        "Cyberpunk Anime",
        "Ghibli-Esque",
        "Dark Fantasy",
        "Kawaii Deco",
        "Retro 90s Anime",
        "Horror Manga"
    ],
    "Digital Painting": [
        "Matte Painting",
        "Character Concept Art",
        "Fantasy Illustration",
        "Hyperreal Portraits",
        "Sci-Fi Concept",
        "Surreal Landscapes",
        "Grimdark Fantasy",
        "Liquid Art",
        "Cosmic Art",
        "Anime-Inspired Painting"
    ],
    "Cyberpunk": [
        "Neon Noir",
        "Retro-Futurism",
        "Biopunk",
        "Cyber Goth",
        "Glitch Art",
        "Holographic UI",
        "Dystopian Cityscapes",
        "Cybernetic Fashion",
        "Neon Samurai",
        "AI Core Aesthetics"
    ],
    "Cinematic Realism": [
        "Film Noir",
        "Blockbuster Action",
        "Thriller Mood",
        "Epic Fantasy",
        "Space Opera",
        "Western Grit",
        "Post-Apocalyptic",
        "Spy Thriller",
        "Period Drama",
        "Monster Horror"
    ],
    "Minimalist Photorealism": [
        "Product Renders",
        "Architectural Visualization",
        "Food Styling",
        "Macro Photography",
        "Fashion Minimalism",
        "Tech Minimalism",
        "Nature Isolation",
        "Portrait Focus",
        "Vehicle Design",
        "Geometric Still Life"
    ],
    "Oil Painting": [
        "Baroque Drama",
        "Romanticism",
        "Impasto Portraits",
        "Dutch Still Life",
        "Abstract Expressionism",
        "Pre-Raphaelite",
        "Cubist Fusion",
        "Modern Portraiture",
        "Wildlife Oils",
        "Mythological Epic"
    ],
    "Watercolor": [
        "Botanical Illustrations",
        "Children’s Book Art",
        "Urban Sketching",
        "Ethereal Portraits",
        "Travel Journal",
        "Marine Art",
        "Seasonal Themes",
        "Abstract Washes",
        "Fantasy Creatures",
        "Minimalist Icons"
    ],
    "Impressionism": [
        "En Plein Air",
        "Floral Abstraction",
        "Urban Impressionism",
        "Seascape Moods",
        "Pointillism",
        "Dance Motion",
        "Seasonal Light",
        "Garden Blooms",
        "Animal Movement",
        "Market Scenes"
    ],
    "Surrealism": [
        "Symbolic Surrealism",
        "Biomechanical",
        "Cosmic Horror",
        "Double Exposure",
        "Impossible Architecture",
        "Surreal Portraits",
        "Dream Collage",
        "Minimalist Surrealism",
        "Whimsical Fantasy",
        "Surreal Geometry"
    ],
    "Pop Art": [
        "Comic Book Pop",
        "Retro Advertising",
        "Celebrity Portraits",
        "Graffiti Fusion",
        "Punk Collage",
        "Kawaii Pop",
        "Op Art",
        "Consumerism Critique",
        "Bold Typography",
        "3D Pop Art"
    ]
}

def apply_fade(image, fade_position='top', fade_ratio=0.4):
    h, w = image.shape[:2]
    fade_height = int(h * fade_ratio)
    mid = h // 2
    if fade_position == 'middle':
        for i in range(fade_height):
            dist_from_mid = abs(i - fade_height // 2)
            alpha = 1 - (dist_from_mid / (fade_height / 2))
            y = mid - fade_height // 2 + i
            if 0 <= y < h:
                image[y] = cv2.addWeighted(image[y], 1 - alpha, np.zeros((w, 3), dtype=np.uint8), alpha, 0)
    elif fade_position == 'top':
        for i in range(fade_height):
            alpha = (fade_height - i) / fade_height
            image[i] = cv2.addWeighted(image[i], 1 - alpha, np.zeros((w, 3), dtype=np.uint8), alpha, 0)
    elif fade_position == 'bottom':
        for i in range(fade_height):
            alpha = i / fade_height
            image[h - fade_height + i] = cv2.addWeighted(image[h - fade_height + i], 1 - alpha, np.zeros((w, 3), dtype=np.uint8), alpha, 0)
    return image

def adjust_font_colors_for_contrast(image, font_color, glow_color, fade_position='top'):
    h, w = image.shape[:2]
    fade_height = int(h * 0.4)
    if fade_position == 'top':
        region = image[:fade_height]
    elif fade_position == 'bottom':
        region = image[h - fade_height:]
    else:
        mid = h // 2
        region = image[mid - fade_height // 2 : mid + fade_height // 2]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    def is_light(color):
        r, g, b = color
        return (0.299*r + 0.587*g + 0.114*b) > 160

    font_is_light = is_light(font_color)
    bg_is_light = mean_brightness > 160

    if font_is_light == bg_is_light:
        font_color = (0, 0, 0) if bg_is_light else (255, 255, 255)

    if abs(np.array(font_color) - np.array(glow_color)).sum() < 60:
        glow_color = (255, 255, 255) if not is_light(font_color) else (0, 0, 0)

    return font_color, glow_color

def add_logo_to_image(base_image, logo_path=r"\Kuku_fm_logo.jpeg", position="top-left", scale=0.1, padding=0):
    """
    Adds a logo to the given base image at a specified position.

    Parameters:
        base_image (np.ndarray): The base image in BGR format (OpenCV).
        logo_path (str): Path to the logo image (supports transparency).
        position (str): One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
        scale (float): Fractional width of base image the logo should take (e.g., 0.1 = 10% width).
        padding (int): Padding in pixels from the edge.

    Returns:
        np.ndarray: Image with logo added (BGR format).
    """
    # Load logo with alpha channel
    logo = Image.open(logo_path).convert("RGBA")
    # Resize logo based on scale
    base_h, base_w = base_image.shape[:2]
    target_w = int(base_w * scale)
    aspect_ratio = logo.height / logo.width
    target_h = int(target_w * aspect_ratio)
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    # Convert base to RGBA
    base_pil = Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Positioning
    positions = {
        "top-left": (padding, padding),
        "top-right": (base_w - target_w - padding, padding),
        "bottom-left": (padding, base_h - target_h - padding),
        "bottom-right": (base_w - target_w - padding, base_h - target_h - padding)
    }
    pos = positions.get(position, (padding, padding))

    # Paste logo with transparency
    base_pil.paste(logo, pos, logo)

    # Convert back to BGR OpenCV image
    result = cv2.cvtColor(np.array(base_pil), cv2.COLOR_RGBA2BGR)
    return result


def draw_title(image, title, font_path, font_size=80, font_color=(255, 255, 255), glow_color=(0, 0, 0), fade_position='top'):
    h, w = image.shape[:2]
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    max_width = int(w * 0.9)
    lines, current_line = [], ""
    words = title.replace(" - ","-").split(" ")
    i = 0
    while i < len(words):
        word = words[i]
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
        i += 1
    if current_line:
        lines.append(current_line)

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    total_height = len(lines) * line_height + (len(lines) - 1) * 10
    y = 20 if fade_position == 'top' else (h - total_height - 20 if fade_position == 'bottom' else (h - total_height) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (w - text_width) // 2
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                draw.text((x + dx, y + dy), line, font=font, fill=glow_color)
        draw.text((x, y), line, font=font, fill=font_color)
        y += line_height + 10

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

st.title("🎨 AI Style & Font Selector with Enhanced Prompts")
title = st.text_input("Enter Title")
concept_note = st.text_area("Enter Concept Note")
name="generated_image"
count=0
output_dir =r"\New folder"

if st.button("Generate 5 Concepts with Styling") and title and concept_note:
    with st.spinner("Generating from OpenAI..."):

        system_prompt = f"""
        You are a branding design assistant helping generate artistic prompts for an AI image generation tool.

        Given a title and concept note:
        1. Choose relevent 2 different artistic styles from the dictionary below which goes with the concept note.
        2. For each style, select 2 sub-styles, totaling *4 outputs*.
        3. Each output must:
        - Be a single JSON object with these fields:
            - "style": the main artistic style
            - "sub_style": the specific sub-genre used
            - "font": one font from this list: {', '.join(available_fonts.keys())}
            - "font_color": HEX color value (e.g., "#ffffff")
            - "glow_color": HEX color value (e.g., "#000000")
            - "prompt": a detailed scene description starting with "Style: Sub-Style", and of strictly 70-80 CLIP tokens, avoid use of title name or character names.  

        4. The prompt should:
        - Begin exactly like: "Style: Sub-Style", (e.g., "Cyberpunk: Neon Noir",)
        - Describe the scene vividly with concrete visual elements (characters, setting, mood, props, atmosphere).
        - Be optimized for AI image generation with rich imagery.
        - Do not repeat the nature of prompts meaning... → Avoid stylistic or thematic repetition; each scene must be visually and narratively distinct.
        - Each must depict a distinctly different scene or environment.
        - You may use up to 80 CLIP tokens; GPT-4 has a tokenizer mismatch, so generate ~50–60 words as a rough guide.

        5. Prompts should be a mixture of portrait type pictures, scenic views, and abstract art. Do not focus exclusively on characters; instead, try to incorporate the underlying theme or essence of the concept note.

        
        Respond only with a valid JSON list in the format:
        [
        {{
            "style": "Cyberpunk",
            "sub_style": "Neon Noir",
            "font": "Orbitron",
            "font_color": "#ffffff",
            "glow_color": "#000000",
            "prompt": "Cyberpunk: Neon Noir, [detailed vivid scene description here...]"
        }},
        ...
        ]

        Art Styles Dictionary:
        {json.dumps(art_styles, indent=2)}
        """
        suggestions = None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\nConcept: {concept_note}"}
                    ]
                )
                suggestions = json.loads(response.choices[0].message.content)

                if isinstance(suggestions, list) and len(suggestions) == 4:
                    required_keys = {"style", "sub_style", "font", "font_color", "glow_color", "prompt"}
                    if all(isinstance(item, dict) and required_keys.issubset(item) for item in suggestions):
                        # Add "path" key to each suggestion after validation
                        for item in suggestions:
                            item["pathtop"] = ""
                            item["pathmiddle"]=""
                            item["pathbottom"]=""# or any default value like a file path string
                        break
                suggestions = None

            except Exception as e:
                st.warning(f"Attempt {attempt+1} failed: {e}")
                suggestions = None

        if not suggestions:
            st.error("Failed to generate 10 valid style prompts after multiple attempts.")
            st.stop()

        for idx, entry in enumerate(suggestions):
            style = entry['style']
            font_name = entry['font']
            font_color = tuple(int(entry['font_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            glow_color = tuple(int(entry['glow_color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            prompt = entry['prompt']
            font_path = rf"C:\Users\Umam\Desktop\fonts\{available_fonts[font_name]}"
            
            sub_style = entry.get("sub_style", "")
            st.markdown(f"### 🔹 Style {idx+1}: {style} — {sub_style}")
            st.markdown(f"Font: {font_name}  |  *Font Color: {entry['font_color']}  |  Glow Color: {entry['glow_color']}*")
            st.code(prompt)
            count=count+1
            path=name+str(count)
            returned_path=playground(prompt,path)
            image = Image.open(returned_path).convert("RGB")
            image_np = np.array(image)[..., ::-1].copy()
            
            for pos in ['top', 'middle', 'bottom']:
                faded = apply_fade(image_np.copy(), fade_position=pos, fade_ratio=0.4)
                font_color_adj, glow_color_adj = adjust_font_colors_for_contrast(
                    faded.copy(), font_color, glow_color, fade_position=pos
                )
                if font_color_adj != font_color or glow_color_adj != glow_color:
                    st.warning("Font or glow color adjusted for better contrast with background.")
                titled = draw_title(
                    faded,
                    title=title,
                    font_path=font_path,
                    font_size=110,
                    font_color=font_color_adj,
                    glow_color=glow_color_adj,
                    fade_position=pos
                )
                final=add_logo_to_image(titled)
                save_path = os.path.join(output_dir, path + pos + ".png")
                Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)).save(save_path)

                entry["path"+pos]=save_path
                st.image(final[..., ::-1], caption=f"{style.title()} - {pos.title()}", channels="RGB")
        score = []
        for idx, entry in enumerate(suggestions):
                image_path1 = entry["pathtop"]
                image_score1 = aesthetic(image_path1)
                image_path2 = entry["pathmiddle"]
                image_score2 = aesthetic(image_path2)
                image_path3 = entry["pathbottom"]
                image_score3 = aesthetic(image_path3)
            
                images = [
                    (image_path1, image_score1),
                    (image_path2, image_score2),
                    (image_path3, image_score3)
                ]
                
                # Sort images based on their score in descending order.
                images_sorted = sorted(images, key=lambda x: x[1], reverse=True)
                
                # Select the top 2 entries.
                top_two = images_sorted[:2]
                
                # Choose one randomly from the top two.
                chosen_image_path, max_score = random.choice(top_two)
                
                score.append({
                    "path": chosen_image_path,
                    "score": max_score,
                    "style": entry['style'],
                    "sub_style": entry.get("sub_style", ""),
                })

                
            
            # Sort all images by score in descending order
        sorted_scores = sorted(score, key=lambda x: x["score"], reverse=True)
            
            # Display ranked images
        st.markdown("## 🏆 Top Ranked Images (All Positions & Styles)")
        for rank, item in enumerate(sorted_scores[:3], start=1):
            image = Image.open(item["path"]).convert("RGB")
            st.image(np.array(image), caption=f"**Rank {rank}**  \nStyle: {item['style']} — {item['sub_style']}  \nScore:{item['score']:.2f}",
                         channels="RGB")

else:
    st.info("Please enter title and concept note")

