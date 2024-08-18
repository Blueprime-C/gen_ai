import glob
import requests
import gradio as gr
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration


MODEL_BASE_URL = 'Salesforce/blip-image-captioning-base'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

processor = AutoProcessor.from_pretrained(MODEL_BASE_URL)
model = BlipForConditionalGeneration.from_pretrained(MODEL_BASE_URL)



def auto_captor(url, file_path, ext, progress=gr.Progress()) -> str:
    progress(0, 'Starting...')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')
    status = []
    with (open(file_path, 'w') as fp):
        for img in progress.tqdm(img_elements):
            img_url = img['src']

            # Correct the image URL if it's malformed
            if img_url.startswith("//"):
                img_url = "https:" + img_url

        # Skip if the image is an SVG or too small (likely an icon) or skip invalid URLs or not in selected extensions
            if 'svg' in img_url or "1x1" in img_url or\
                not (img_url.startswith('http://') or img_url.startswith('https://')) or \
                    not img_url.split('.')[-1] in ext:
                continue

            try:
                response = requests.get(img_url)
                raw_image = Image.open(BytesIO(response.content))
                if raw_image.size[0] * raw_image.size[1] < 400: # Skip very small images
                    continue

                raw_image = raw_image.convert('RGB')
                inputs = processor(raw_image, return_tensors='pt')
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                fp.write("%s: %s\n" % (img_url, caption))
            except Exception as e:
                status.append("Error processing image %s: %s" % (img_url, str(e)))
    return not status and "Success" or "Error: \n %s" % '\n'.join(errors for errors in status)

UI = gr.Interface(
    fn=auto_captor,
    inputs=[
        "text",
        gr.File( file_count="single", file_types=[".txt"]),
        gr.CheckboxGroup(ALLOWED_EXTENSIONS, label='Image Extensions', info="Select extension to track")],
    outputs="text",
    title="Automated Image Captioning Tool",
    description="Enter an valid url for automatic image captioning.",
    example_labels=["https://example.com"]
).queue()


if __name__ == '__main__':
    UI.launch(share=True)