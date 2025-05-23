import clip
import torch
from PIL import Image
import numpy as np
import gensim.downloader as api

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Candidate captions for zero-shot image understanding
candidate_captions = [
    "a man riding a horse"
]

def get_clip_caption(image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(candidate_captions).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
        logits_per_image, _ = clip_model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    best_caption = candidate_captions[np.argmax(probs)]
    return best_caption

def get_fasttext_vector(text):
    words = text.lower().split()
    vectors = [fasttext_model[word] for word in words if word in fasttext_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def score_similarity(text1, text2):
    v1 = get_fasttext_vector(text1)
    v2 = get_fasttext_vector(text2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


image_path = "Input/Image/man_horse.jpg"  # Replace with your image
user_description = "a person riding a horse in the field"

generated_caption = get_clip_caption(image_path)
score = score_similarity(generated_caption, user_description)

print("Generated Caption:", generated_caption)
print("User Description:", user_description)
print("Similarity Score:", round(score, 2))
