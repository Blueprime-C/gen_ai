import glob
import os
import requests
import gradio as gr
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

from image_captioning.intro import img_url

MODEL_BASE_URL = 'Salesforce/blip-image-captioning-base'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
DEFAULT_CUT_OFF = '400'

processor = AutoProcessor.from_pretrained(MODEL_BASE_URL)
model = BlipForConditionalGeneration.from_pretrained(MODEL_BASE_URL)


def image_generator(url, ext:str, progress):
    if url.startswith('http'):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_elements = soup.find_all('img')
        for img in progress.tqdm(img_elements):
            img_url = img['src']
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            # Skip if the image is an SVG or too small (likely an icon) or skip invalid URLs or not in selected extensions
            if not ('svg' in img_url or "1x1" in img_url) and img_url.startswith('http') and img_url.split('.')[-1] in ext:
                response = requests.get(img_url)
                yield img_url, Image.open(BytesIO(response.content))
    else:
        for _ext in ext:
            for paths in progress.tqdm(glob.glob(os.path.join(url, f'*.{_ext}'))):
                yield paths, Image.open(paths)

def auto_captor(url:str, file_path:str, ext:str, cut_off:str=DEFAULT_CUT_OFF, progress=gr.Progress()) -> str:
    progress(0, 'Starting...')

    status = []
    with (open(file_path, 'w') as fp):
        for img_path, image in image_generator(url, ext, progress):

            try:

                if image.size[0] * image.size[1] < int(cut_off): # Skip images which resolution is less than cut-off
                    continue

                image = image.convert('RGB')
                inputs = processor(image, return_tensors='pt')
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                fp.write("%s: %s\n" % (img_path, caption))
            except Exception as e:
                status.append("Error processing image %s: %s" % (img_path, str(e)))
    return not status and "Success" or "Error: \n %s" % '\n'.join(errors for errors in status)

UI = gr.Interface(
    fn=auto_captor,
    inputs=[
        "text",
        gr.File( file_count="single", file_types=[".txt"]),
        gr.CheckboxGroup(ALLOWED_EXTENSIONS, label='Image Extensions', info="Select extension to track"),
        gr.Text(lines=1, placeholder="Resolution Cut-off(integer) of images; Default is 400px", value='400'),
    ],
    outputs="text",
    title="Automated Image Captioning Tool",
    description="Enter an valid url for automatic image captioning.",
    example_labels=["https://example.com"]
).queue()


if __name__ == '__main__':
    UI.launch(share=True)