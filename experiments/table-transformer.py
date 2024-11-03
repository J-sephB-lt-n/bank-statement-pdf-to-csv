"""
Identify table in image using Microsoft table-transformer

usage:
    $ python experiments/table-transformer.py "/path/to/img.png"
"""

import torch
from transformers import AutoImageProcessor, TableTransformerModel, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import sys

# file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
file_path = sys.argv[1] #"/Users/josephbolton/Downloads/bank_statement_images/output-1.png"
image = Image.open(file_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-detection"
)
print("loaded image processor")
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)
print("loaded model")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes
)[0]


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    draw = ImageDraw.Draw(image)
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    xmin, ymin, xmax, ymax = box
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    image.show()
