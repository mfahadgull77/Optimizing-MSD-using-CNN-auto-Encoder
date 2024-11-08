import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


input_dir = ""

output_dir = ""
os.makedirs(output_dir, exist_ok=True)

def process_image(input_path, output_image_path, output_histogram_path):
    
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Error loading image: {input_path}")
        return
    

    resized_image = cv2.resize(image, (512, 512))


    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

  
    brightened_image = cv2.add(gray_image, brightness_increase)
    brightened_image = np.clip(brightened_image, 0, 255)  # Clip values to the range [0, 255]


    min_intensity = np.min(brightened_image)
    max_intensity = np.max(brightened_image)
    contrast_adjusted_image = (brightened_image - min_intensity) / (max_intensity - min_intensity) * 255
    contrast_adjusted_image = contrast_adjusted_image.astype(np.uint8)


    cv2.imwrite(output_image_path, contrast_adjusted_image)


    plt.figure(figsize=(8, 6))
    plt.hist(contrast_adjusted_image.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram of Contrast-Adjusted Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.savefig(output_histogram_path, format='jpeg', dpi=dpi)
    plt.close()
    print(f"Processed and saved image and histogram: {output_image_path}, {output_histogram_path}")


for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)
        output_histogram_path = os.path.join(output_dir, filename.replace('.png', '_hist.jpg').replace('.jpg', '_hist.jpg').replace('.jpeg', '_hist.jpg'))
        process_image(input_path, output_image_path, output_histogram_path)
