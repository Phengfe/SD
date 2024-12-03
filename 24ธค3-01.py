# ติดตั้งไลบรารีที่จำเป็น
!pip install gradio diffusers transformers torch torchvision torchaudio accelerate
!pip install pillow

# --- นำเข้าไลบรารี ---
import os
import random
import torch
from diffusers import StableDiffusionPipeline
from google.colab import drive
from PIL import Image
import gradio as gr

# --- ฟังก์ชันต่าง ๆ ---
def mount_drive():
    """
    เชื่อม Google Drive
    """
    drive.mount('/content/drive')
    return "Google Drive Mounted Successfully."

def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    """
    โหลดโมเดล Stable Diffusion
    """
    print(f"Loading model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
    print("Model loaded successfully.")
    return pipe

def generate_images(prompt, width, height, num_images, seed, steps, guidance_scale, model):
    """
    สร้างภาพจาก Stable Diffusion
    """
    if model is None:
        return "Error: Model not loaded!", []

    # กำหนด Seed
    if seed == -1:  # สุ่ม seed หากไม่ระบุ
        seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)

    # สร้างภาพ
    images = model(prompt, 
                   width=width, 
                   height=height, 
                   num_inference_steps=steps, 
                   guidance_scale=guidance_scale, 
                   num_images_per_prompt=num_images, 
                   generator=generator).images
    return images, seed

def save_images_to_drive(images, seed, folder_name="StableDiffusion"):
    """
    บันทึกภาพไปยัง Google Drive
    """
    folder_path = f"/content/drive/MyDrive/{folder_name}/"
    os.makedirs(folder_path, exist_ok=True)
    save_paths = []
    for idx, img in enumerate(images):
        output_path = os.path.join(folder_path, f"output_{seed}_{idx+1}.png")
        img.save(output_path)
        save_paths.append(output_path)
    return save_paths

# --- โหลดโมเดล ---
try:
    pipe = load_model()
except Exception as e:
    pipe = None
    print(f"Error loading model: {e}")

# --- UI Gradio ---
def process(prompt, width, height, num_images, seed, steps, guidance_scale, folder_name):
    """
    ดำเนินการสร้างภาพและบันทึกใน Google Drive
    """
    if pipe is None:
        return "Error: Model not loaded!", []

    # สร้างภาพ
    images, used_seed = generate_images(prompt, width, height, num_images, seed, steps, guidance_scale, model=pipe)

    # บันทึกภาพใน Google Drive
    save_paths = save_images_to_drive(images, used_seed, folder_name)

    return f"Images saved to folder: {folder_name}", images

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Image Generator with Advanced Options")

    # Input: Prompt
    prompt_input = gr.Textbox(label="Enter your Prompt", placeholder="Describe the image you want to create...")

    # Advanced Inputs
    width_input = gr.Slider(128, 1024, value=512, step=128, label="Image Width")
    height_input = gr.Slider(128, 1024, value=512, step=128, label="Image Height")
    num_images_input = gr.Slider(1, 5, value=1, step=1, label="Number of Images")
    seed_input = gr.Number(value=-1, label="Random Seed (-1 for random)")
    steps_input = gr.Slider(10, 100, value=50, step=10, label="Inference Steps")
    guidance_input = gr.Slider(5, 15, value=8.0, step=0.5, label="Guidance Scale")
    folder_input = gr.Textbox(value="StableDiffusion", label="Output Folder in Google Drive")

    # Output: Save Path and Gallery
    save_output = gr.Textbox(label="Save Path")
    image_output = gr.Gallery(label="Generated Images")  # ไม่มี .style()

    # Button to generate
    generate_button = gr.Button("Generate and Save Images")

    # Event: Process
    generate_button.click(
        process, 
        inputs=[prompt_input, width_input, height_input, num_images_input, seed_input, steps_input, guidance_input, folder_input],
        outputs=[save_output, image_output]
    )

# เปิด UI
demo.launch()
