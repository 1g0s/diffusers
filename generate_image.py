from diffusers import DiffusionPipeline
import torch
import os

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")  # Use CUDA if available

# Generate an image
prompt = "A photo of New York"
images = pipeline(prompt).images

# Ensure output directory exists
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Find the next available index for image saving
existing_files = os.listdir(output_dir)
next_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith('generated_image_')], default=-1) + 1

# Save the images
for i, image in enumerate(images):
    image.save(os.path.join(output_dir, f"generated_image_{next_index + i}.png"))

# Display the first image (optional)
images[0].show()