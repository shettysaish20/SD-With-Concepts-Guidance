import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import gradio as gr
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA for Gradio
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_home"

# =====================================================
# Configuration: CPU only and Stable Diffusion v1-4
# =====================================================
torch_device = "cpu"
print(f"Using device: {torch_device}")

# Load the Stable Diffusion v1-4 pipeline on CPU with float32
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    use_auth_token=True  # make sure you have a valid Hugging Face token
).to(torch_device)

# Load and attach the safety checker
pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float32
).to(torch_device)

# Setup components for easier access
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Model configuration parameters
batch_size = 1
height = 512
width = 512
guidance_scale = 6.0
num_inference_steps = 50
# Set a default value for airplane_guidance_scale; 0 means no airplane guidance
default_airplane_guidance_scale = 100  

# =====================================================
# Text embedding helper components and functions
# =====================================================
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
position_embeddings = text_encoder.text_model.embeddings.position_embedding.weight

def get_output_embeds(input_embeddings):
    """Process input embeddings through the text encoder to get output embeddings"""
    with torch.no_grad():
        outputs = text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=torch.ones(
                input_embeddings.shape[0], input_embeddings.shape[1], 
                dtype=torch.float32, device=torch_device
            ),
        )
        return text_encoder.text_model.final_layer_norm(outputs[0])

# =====================================================
# Load or define concept embeddings
# =====================================================
# Dictionary to store concept embeddings with their respective tokens
concept_embeddings = {}
concepts = ["<anime-background-style-v2>", "<gta5-artwork>", "<line-art>", "<m-geo>", "<orientalist-art>"]  # Replace with your 5 concepts

# Load each concept embedding
for concept in concepts:
    concept_path = f"sd-concepts/{concept.strip('<>')}-learned_embeds.bin"  # Adjust path as needed
    if os.path.exists(concept_path):
        embed_dict = torch.load(concept_path, map_location=torch_device, weights_only=False)
        concept_token = list(embed_dict.keys())[0]  # Get the token
        concept_embeddings[concept] = embed_dict[concept_token]
        print(f"Loaded concept: {concept}")
    else:
        print(f"Warning: Embedding file for {concept} not found at {concept_path}")

# # =============================================
# PART 2: Extract latent embeddings from airplane images
# =============================================

def extract_vae_latents(image_paths):
    """Extract VAE latent embeddings from a list of images."""
    latents_list = []
    
    for img_path in tqdm(image_paths, desc="Extracting latents"):
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((width, height))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(torch_device)  # [1, 3, H, W]
        image_tensor = 2 * image_tensor - 1  # Scale to [-1, 1]
        
        # Extract latent through VAE encoder
        with torch.no_grad():
            latent = vae.encode(image_tensor.to(dtype=vae.dtype)).latent_dist.sample() * 0.18215
        
        latents_list.append(latent)
    
    # Stack all latents
    if latents_list:
        all_latents = torch.cat(latents_list, dim=0)
        # Calculate average latent
        avg_latent = all_latents.mean(dim=0, keepdim=True)
        return avg_latent
    else:
        return None

# Path to your generated airplane images
airplane_image_paths = [f"clip_images/plane_{i}.png" for i in range(10)]

# Extract average latent embedding from airplane images
print("Extracting latent embeddings from airplane images...")
airplane_latent = extract_vae_latents(airplane_image_paths)
print("Airplane latent shape:", airplane_latent.shape if airplane_latent is not None else "None")

# =====================================================
# Inference function: Generate image with concept & guidance
# =====================================================
def generate_with_concept_and_guidance(prompt_base, concept_token, seed, airplane_guidance_scale):
    """
    Generate an image using a concept token and optional airplane guidance.
    """
    # Set the random seed for reproducibility
    generator = torch.manual_seed(seed)
    
    # Use a custom placeholder token "<my_concept>" in the prompt
    prompt = f"{prompt_base} <my_concept>"
    print(f"Generating: {prompt} (replacing <my_concept> with {concept_token})")
    
    # Tokenize the prompt
    text_input = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = text_input.input_ids.to(torch_device)
    
    # Get token embeddings for the prompt
    token_embeddings = token_emb_layer(input_ids)
    
    # Find the placeholder token "<my_concept>" in the input IDs
    placeholder_token_id = tokenizer.encode("<my_concept>", add_special_tokens=False)[0]
    placeholder_indices = (input_ids[0] == placeholder_token_id).nonzero(as_tuple=True)[0]
    
    if len(placeholder_indices) > 0 and concept_token in concept_embeddings:
        # Replace the placeholder embedding with the concept embedding
        concept_embed = concept_embeddings[concept_token].to(torch_device)
        for idx in placeholder_indices:
            token_embeddings[0, idx] = concept_embed
        print(f"Replaced <my_concept> token with {concept_token} embedding")
    else:
        print(f"Warning: Placeholder token not found or {concept_token} embedding not available.")
    
    # Add position embeddings and get the final text embeddings
    position_ids = torch.arange(0, input_ids.shape[1], device=torch_device).unsqueeze(0)
    position_embs = position_embeddings[position_ids]
    input_embeddings = token_embeddings + position_embs
    text_embeddings = get_output_embeds(input_embeddings)
    
    # Unconditional embeddings for classifier-free guidance with a negative prompt
    uncond_input = tokenizer(
        ["nude, naked, explicit, porn, sexual, NSFW"] * batch_size, 
        padding="max_length", 
        max_length=text_input.input_ids.shape[1], 
        return_tensors="pt"
    ).input_ids.to(torch_device)
    uncond_embeddings = text_encoder(uncond_input)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(torch_device)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop with optional airplane guidance
    for i, t in enumerate(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i] if hasattr(scheduler, "sigmas") else 0
        latent_model_input = scheduler.scale_model_input(latent_model_input, t).to(torch.float32)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings.to(torch.float32)).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Apply airplane guidance every 5 timesteps if airplane_latent is provided
        if i % 5 == 0 and airplane_latent is not None:
            latents = latents.detach().requires_grad_()
            latents_x0 = latents - sigma * noise_pred if sigma > 0 else latents
            # Using cosine similarity as the loss (multiplied by the guidance scale)
            loss = F.cosine_similarity(latents_x0.view(latents_x0.shape[0], -1),
                                       airplane_latent.view(airplane_latent.shape[0], -1), dim=1).mean() * airplane_guidance_scale
            print(f"Step {i}, airplane guidance loss: {loss.item()}")
            cond_grad = torch.autograd.grad(loss, latents)[0]
            latents = latents.detach() - cond_grad * (sigma**2 if sigma > 0 else 1.0)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode the latents to an image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
    
    # Post-process the image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    
    # (Optional) Run safety checker if desired – here we omit it in the inference loop.
    return image

# =====================================================
# Gradio Interface
# =====================================================
def inference(prompt_base, concept_token, seed, airplane_guidance_scale):
    return generate_with_concept_and_guidance(prompt_base, concept_token, seed, airplane_guidance_scale)

iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(label="Prompt Base", value="A person walking in the beach"),
        gr.Dropdown(label="Concept Token", choices=concepts, value="<gta5-artwork>"),
        gr.Slider(minimum=0, maximum=1000, step=1, label="Seed", value=42),
        gr.Slider(minimum=0, maximum=500, step=1, label="Airplane Guidance Scale", value=0)
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Stable Diffusion with Concept & Airplane Guidance (CPU)",
    description="Generate images using Stable Diffusion v1-4 with injected concept and optional airplane guidance. Running on CPU."
)

iface.launch()
