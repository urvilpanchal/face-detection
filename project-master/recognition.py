import cv2
import os
import face_recognition
import pickle
import numpy as np

print(" Loading Encode File...")


try:
    with open('EncodeFile.p', 'rb') as file:
        existing_encoding, existing_id = pickle.load(file)
    print("Encode File Loaded Successfully!")
except (FileNotFoundError, EOFError):
    print("Error: Encode file not found or is empty!")
    exit()


given_id = "IMG_20241016_215024"


if given_id not in existing_id:
    print(f" Error: Given ID '{given_id}' not found in stored encodings!")
    exit()


index = existing_id.index(given_id)
encoding = [existing_encoding[index]]
print(f"🔍 Encoding found for ID: {given_id}")


path_to_images = r"D:\photos"

# Check if the directory exists
if not os.path.exists(path_to_images):
    print(f"Error: The folder '{path_to_images}' does not exist.")
    exit()


image_files = [f for f in os.listdir(path_to_images) if f.endswith(('jpg', 'jpeg', 'png'))]


RECOGNITION_THRESHOLD = 0.50  

recognized_images = []


for img_name in image_files:
    img_path = os.path.join(path_to_images, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Could not read '{img_name}', skipping.")
        continue

    
    imgS = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

   
    faceLoc = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceLoc)

    if not faceLoc or not encodeCurFrame:
        print(f"No face detected in '{img_name}', skipping.")
        continue

    
    for encodeFace, facePosition in zip(encodeCurFrame, faceLoc):
        faceDis = face_recognition.face_distance(encoding, encodeFace)
        
        if len(faceDis) == 0:  
            continue

        matchIndex = np.argmin(faceDis)  
        print(f" Face Distance for '{img_name}': {faceDis[matchIndex]:.4f}")

        if faceDis[matchIndex] < RECOGNITION_THRESHOLD:
            recognized_images.append(img_name)
            print(f" Face recognized in '{img_name}' for ID: {given_id}")


if recognized_images:
    print(" Recognized Images:", recognized_images)
else:
    print("No matching faces found.")
