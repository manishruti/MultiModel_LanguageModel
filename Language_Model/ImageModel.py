# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import logging
logging.set_verbosity_error()


from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import config

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import threading
import time

def show_image_auto_close(image_path, delay=5):
    img = mpimg.imread(image_path)

    def close_after_delay():
        time.sleep(delay)
        plt.close()

    # Start a timer in a separate thread
    threading.Thread(target=close_after_delay, daemon=True).start()

    # Show image (this blocks until plt.close() is called)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image will close in {delay} seconds...")
    plt.show()


# Load BLIP model for image captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
caption_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Sentence Transformer model for scoring
similarity_model = SentenceTransformer(config.multilingualTransformer)

def generate_caption(image_path):
    image  = Image.open(image_path).convert('RGB')
    inputs = caption_processor(image, return_tensors="pt")
    out    = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def score_description(true_caption, user_caption):
    emb1 = similarity_model.encode(true_caption, convert_to_tensor=True)
    emb2 = similarity_model.encode(user_caption, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)

if __name__ == "__main__":

    results = []
    image_paths = []
    IMAGE_DIR = 'Input/Image/'

    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            path = os.path.join(IMAGE_DIR, filename)
            image_paths.append((filename, path))

            print(f"\n Showing image: {filename}")
            show_image_auto_close(path, delay=10)

            user_input = input(" Describe the image: ").strip()
            results.append((filename, path, user_input))

    # -------- CAPTION GENERATION --------
    final_results = []
    for filename, path, user_input in results:
        model_caption = generate_caption(path)
        final_results.append((filename, model_caption, user_input))

    # -------- SCORING --------

    print("\n Summary of Results:")
    for filename, model_caption, user_input in final_results:
        score = score_description(model_caption, user_input)
        print(f"\n {filename}")
        print(f"Model Caption : {model_caption}")
        print(f"Your Caption  : {user_input}")
        print(f"Similarity Score: {round(score, 2)}")
