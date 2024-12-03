# ติดตั้งไลบรารีที่จำเป็น
!pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display
from PIL import Image

# โหลดโมเดล
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# สร้างภาพ
prompt = "A beautiful landscape with mountains and rivers"
image = pipe(prompt).images[0]

# บันทึกภาพ
image.save("output.png")

# แสดงผลใน Colab
display(image)
