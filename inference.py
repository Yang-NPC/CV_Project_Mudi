import os

from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection

from adapter_modules.adapter import PositionNetPLUS as PositionNet
from adapter_modules.model import MuseAdapter
from adapter_modules.utils import save_generated_images, seed_everything


class MUSE_pipeline(torch.nn.Module):
    def __init__(self, muse_ckpt, dynamic_scale=False, scale_schedule="linear"):
        super().__init__()

        #https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
        base_model_path = "/home/zchengay/Model/sdxl-base"
        # https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
        image_encoder_path = "/home/zchengay/Model/CLIP-G"
        
        self.device = "cuda"
        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=torch.float16)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.pipe.to(self.device)

        num_tokens_image = 4
        num_tokens_text = 1

        self.image_proj_model = PositionNet(in_dim_text=768, in_dim_image=self.image_encoder.config.hidden_size,
                out_dim_text=2048, out_dim_image=self.pipe.unet.config.cross_attention_dim, fourier_freqs=16, num_tokens_text=num_tokens_text, num_tokens_image = num_tokens_image
                ).to(self.device, dtype=torch.float16)

        self.muse_model = MuseAdapter(self.pipe.unet, self.image_proj_model, ckpt_path=muse_ckpt, device=self.device, num_tokens=num_tokens, dynamic_scale=dynamic_scale, scale_schedule=scale_schedule)

    def generate(
        self,
        imgs=["./examples/example_dog.jpg", "./examples/example_cat.jpg"], 
        phrases = [["dog", "cat"]], 
        boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]], 
        prompt = "best quality, high quality, a dog and cat on the beach", 
        save_img_name = None,
        num_samples=4, 
        result_path = "./res",
        log_id = 'test',
        height = 512,
        width = 512,
        max_box_num = None,
        scale =1.0,
        seed = 10626,
        num_inference_steps=30,
        debug_predict_step=None,
        debug_save_dir=None,
        ):
        
        input_images = [Image.open(img) for img in imgs]
        input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

        for i in range(max_box_num - len(phrases[0])):
            phrases[0].append("")

        masks= torch.zeros([1, max_box_num]).to(self.device, dtype = torch.float16)
        masks[0, :len(imgs)] =1

        text_masks = masks
        image_masks = masks

        images = self.muse_model.generate(pipe=self.pipe, num_samples=num_samples, num_inference_steps=num_inference_steps, seed=seed,
                                prompt=[prompt], scale=scale, boxes=boxes, phrases=phrases, height=height, width=width ,
                                max_box_num = max_box_num, masks=masks, text_masks=text_masks, 
                                pil_images=input_images, image_masks=image_masks, image_encoder=self.image_encoder,
                    image_processor=self.image_processor,
                    debug_predict_step=debug_predict_step,
                    debug_save_dir=debug_save_dir,
                                )

        if result_path and log_id:
            save_path = os.path.join(result_path, log_id)
            save_generated_images(images, boxes, phrases, save_path, save_img_name, prompt, width, height)
        return images
if __name__ == "__main__":
    muse_ckpt = "/home/zchengay/Model/MUSE/muse_weight.pth"
    num_tokens = 4
    log_id = f'inference'
    result_path="/home/zchengay/CV/MUSE/result"
    
    seed_everything(0)

    # Common parameters for all schedules
    #imgs = [ "/home/zchengay/CV/MUSE/imgs/dog.jpg", "/home/zchengay/CV/MUSE/imgs/cat.jpg", "/home/zchengay/CV/MUSE/imgs/hat.png"]
    #phrases = [["dog", "cat", "hat"]]
    #boxes = [[[0.3, 0.5, 0.5, 0.8], [0.5, 0.6, 0.7, 0.8], [0.3, 0.4, 0.6, 0.5]]]

    # prompt = "a dog wearing the hat and sitting next to one cat in the garden "
    height = 1024
    width = 1024
    imgs = [ "/home/zchengay/CV/MUSE/imgs/dog1.jpg", "/home/zchengay/CV/MUSE/imgs/dog2.jpg", "/home/zchengay/CV/MUSE/imgs/cat1.jpg",]
    phrases = [["dog", "dog", "cat"]]
    boxes = [[[0.1, 0.1, 0.55, 0.6], [0.45, 0.3, 0.75, 0.6], [0.7, 0.4, 0.9, 0.6]]]
    prompt = "a cat , a dog and a dog they are sitting in the garden."

    # All available schedules
    schedules = ["static", "linear", "cosine", "exponential", "reverse_linear", "increasing", "decreasing"]
    t_steps = [1, 5, 10, 15, 20, 25, 30]
    for schedule, num_step in zip(schedules, t_steps):
        print(f"\n{'='*50}")
        print(f"Generating image with schedule: {schedule}")
        print(f"{'='*50}")
        
        # Create pipeline with current schedule
        pipeline = MUSE_pipeline(muse_ckpt, dynamic_scale=True, scale_schedule=schedule)
        
        save_img_name = f"1_{schedule}_dog_dog_cat_1024"
        
        images = pipeline.generate(
            imgs=imgs,
            phrases=phrases,
            boxes=boxes,
            prompt=prompt,
            num_samples=1,
            log_id=log_id,
            result_path=result_path,
            height=height,
            width=width,
            max_box_num=10,
            save_img_name=save_img_name,
            scale=0.8,
            debug_predict_step=num_step,
            debug_save_dir=os.path.join(result_path, log_id, "debug", schedule),
        )
        
        print(f"Saved: {save_img_name}")
        
        # Clean up to free GPU memory
        del pipeline
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("All schedules completed!")
    print(f"{'='*50}")

