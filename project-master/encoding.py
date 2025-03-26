import cv2
import face_recognition
import os
import pickle
import concurrent.futures
import numpy as np


path_to_images = r"D:\photos"  
if not os.path.exists(path_to_images):
    print(f"Error: The folder '{path_to_images}' does not exist.")
    exit()

try:
    with open('EncodeFile.p', 'rb') as file:
        existing_encodings, existing_ids = pickle.load(file)
    print("Loaded existing encodings.")
except (FileNotFoundError, EOFError):
    existing_encodings, existing_ids = [], []
    print(" No existing encodings found, starting fresh.")


image_files = [f for f in os.listdir(path_to_images) if f.endswith(('jpg', 'jpeg', 'png'))]
new_images, new_ids = [], []


for img_name in image_files:
    id = os.path.splitext(img_name)[0]
    if id in existing_ids:
        print(f"Skipping '{id}', already encoded.")
        continue  

    img_path = os.path.join(path_to_images, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f" Warning: Could not read '{img_name}', skipping.")
        continue

    new_images.append(img)
    new_ids.append(id)

print(f" Found {len(new_images)} new images to encode.")


def encode_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)
    return encodings[0] if encodings else None  


if new_images:
    print(" Generating encodings for new images...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_encodings = list(executor.map(encode_image, new_images))

    
    valid_encodings = [enc for enc in new_encodings if enc is not None]
    valid_ids = [new_ids[i] for i in range(len(new_encodings)) if new_encodings[i] is not None]

    
    existing_encodings.extend(valid_encodings)
    existing_ids.extend(valid_ids)

   
    with open('EncodeFile.p', 'wb') as file:
        pickle.dump([existing_encodings, existing_ids], file)

    print(f" Successfully saved {len(valid_encodings)} new encodings!")

else:
    print("No new images to encode.")

