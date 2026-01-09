from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

image = load_image(
    "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/ladysprite.png"
)
org_size = image.size
image = openpose(image)
image.save("/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/ladysprite_open_pose.png")

controlnet = ControlNetModel.from_single_file(
    "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/controlnets/control_v11p_sd15_openpose_fp16.safetensors", 
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_single_file(
    "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors", 
    controlnet=controlnet, 
    safety_checker=None, 
    feature_extractor=None,
    torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

width,height = org_size

image = pipe(
    "Horny slut spreading her legs", 
    image, 
    num_inference_steps=45, 
    guidance_scale=8.5, 
    strength=4.4 ,
    height=height,
    width=width
).images[0]

image.save('/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/canny_images/chef_pose_out.png')
