from prompt import FastSAMPrompt
from ultralytics import FastSAM
from PIL import Image

# Define an inference source
source = "Your image path"

# Create a FastSAM model
model = FastSAM("Pretrained model weight path")

# Run inference on an image
everything_results = model(source, device="cuda", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

# visualize and save
input = Image.open(source)
input = input.convert("RGB")

prompt_process = FastSAMPrompt(input, everything_results, device="cuda")
ann = prompt_process.everything_prompt()

prompt_process.plot(
    annotations=ann,
    output_path="./output/"+source.split("/")[-1],
    bboxes = None,
    points = None,
    point_label = None,
    withContours=False,
    better_quality=False,
)