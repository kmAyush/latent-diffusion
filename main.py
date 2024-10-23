import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"
ALLOW_CUDA  = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA: 
    DEVICE = "cuda"
# elif torch.backends.mps.is_built() and ALOW_MPS==True:
#     DEVICE = "mps"

print(f"Using device {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

print("Model loaded")

# Text to Image
prompt = "A dog on a beach"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7

# Image to Image 
input_image = None
image_path = "images/dog.jpg"
strength = 0.9

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt              = prompt,
    uncond_prompt       = uncond_prompt,
    input_image         = input_image,
    strength            = strength,
    do_cfg              = do_cfg,
    cfg_scale           = cfg_scale,
    sampler_name        = sampler,
    n_inference_steps   = num_inference_steps,
    seed                = seed,
    models              = models,
    device              = DEVICE,
    idle_device         = "cpu",
    tokenizer           = tokenizer

)

img = Image.fromarray(output_image)
img.show()