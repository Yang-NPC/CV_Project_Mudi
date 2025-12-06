# Dynamic Attention Scheduling for Multi-Subject Identity-Preserving Generation with Automatic Pipeline
This is the repository for our CV final project, we introduce dynamic attention scheduling and automatic pipeline to further improve the multi-subject generation quality based on the MUSE repo.


## Visual Results
the visual results are in our technical report. Also the reuslt folder contains some of the results.

## Requirement
```
conda create -n DAS python=3.9.25
conda activate DAS
pip install -r requirements.txt
```
## Model Preparation
1. **Download Base Models**: Download the pretrained [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [CLIP-G](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) models.
2. **Download MUSE Checkpoint**: Download our [MUSE](https://huggingface.co/pf0607/MUSE) model checkpoint.
2. **Download Qwen 2.5 vl 7B Checkpoint**: Download our [Qwen 2.5 vl 7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) model checkpoint.
## Inference
use inference.py for inference to see our dynamic attention scheduling results.
use auto_layout_pipeline.py for automatic layout pipeline inference.

**2 Subjects Inference**:
```
python inference.py
```
**3 Auto pipeline Inference**:
```
python auto_layout_pipeline.py
```

