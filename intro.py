from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# img = Image.open('./data/images/img_1.png')
# inputs = processor(img, return_tensors="pt")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
text = "what is complexion of the woman?"
inputs = processor(image, text, return_tensors="pt")

outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption: ", caption)