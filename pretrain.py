import cv2, os
from src.anti_spoof_predict import AntiSpoofPredict
from PIL import Image

def scale_and_crop(bbox, original_image_size):
    left, top, width, height = bbox
    target_ratio = 1 / 1
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = height * target_ratio
        new_height = height
    else:
        new_width = width
        new_height = width / target_ratio
    
    scale_factor = 1
    
    new_width *= scale_factor
    new_height *= scale_factor
    
    original_width, original_height = original_image_size
    
    if left + new_width > original_width:
        new_width = original_width - left
    if top + new_height > original_height:
        new_height = original_height - top
    if left < 0:
        new_width += left
        left = 0
    if top < 0:
        new_height += top
        top = 0
    
    left = max(left - (new_width - width) / 2, 0)
    top = max(top - (new_height - height) / 2, 0)
    
    return [int(left), int(top), int(new_width), int(new_height)]


def resize_image(array_image, target_width = 80, target_height = 80):
    array_image = cv2.cvtColor(array_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(array_image)
    resized_image = image.resize((target_width, target_height))
    return resized_image


model_test = AntiSpoofPredict(0)

# SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'images', 'sample', 'fake')
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'images', 'sample-training')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'images', 'output-pretrain')

files = os.listdir(SAMPLE_PATH)
counter = 1

for file in files:
    print(f"{counter}/{len(files)}: Processing image {file} ... ")

    image = cv2.imread(os.path.join(SAMPLE_PATH, file))
    
    image_bbox = model_test.get_bbox(image) # [left, top, width, height]
    new_bbox = scale_and_crop(image_bbox, (image.shape[1], image.shape[0])) # [left, top, width, height]
    cropped_image = image[new_bbox[1]:(new_bbox[1]+new_bbox[3]), new_bbox[0]:(new_bbox[0]+new_bbox[2])]
    resized_image = resize_image(cropped_image)
    resized_image.save(os.path.join(OUTPUT_PATH, file))

    counter+=1
