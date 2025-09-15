# Diffusers Library for Quickly Building and Deploying Diffusion Models

Diffusers is an open-source library developed by Hugging Face that aims to simplify the construction and deployment of diffusion models. It provides pre-trained diffusion models, easy-to-use APIs, and seamless integration with the Hugging Face ecosystem, enabling researchers and developers to quickly implement and apply diffusion models.

Official Docs: https://huggingface.co/docs/diffusers/index

## Unique Features
### Scheduler
A scheduler in diffusion models controls how noise is gradually removed from an image during the generation process.

When generating an image, the model starts from pure noise and goes through many steps to produce a clear image. In each step, the model predicts the noise in the current image, and the scheduler decides how much of that noise to remove and how to update the image for the next step.

#### Timestep
Controls the number of steps taken to generate an image. More timesteps can lead to higher quality images but take longer to compute. Only used during inference.

```
from diffusers.schedulers import AysSchedules

sampling_schedule = AysSchedules["StableDiffusionXLTimesteps"]
print(sampling_schedule)
"[999, 845, 730, 587, 443, 310, 193, 116, 53, 13]"
```

#### Sigmas
A list of noise levels corresponding to each timestep. Higher sigma values indicate more noise, while lower values indicate less noise. The model uses these values to determine how much noise to remove at each step.
```
import torch

from diffusers import DiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16,
  variant="fp16",
).to("cuda")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0]
prompt = "anthropomorphic capybara wearing a suit and working with a computer"
generator = torch.Generator(device='cuda').manual_seed(123)
image = pipeline(
    prompt=prompt,
    num_inference_steps=10,
    sigmas=sigmas,
    generator=generator
).images[0]
```
