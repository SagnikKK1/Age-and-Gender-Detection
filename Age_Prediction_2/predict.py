import cv2
import torch
from facenet_pytorch import MTCNN
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dropout, Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--type', default="video",
                    help='type of input ')

args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

def preprocess_image(image, target_size=(200, 200)):
    # Normalize pixel values to range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Resize the image
    image = tf.image.resize(image, target_size)

    # Convert to grayscale
    # image = tf.image.rgb_to_grayscale(image)

    # Remove blurriness and noise using Gaussian blur
    image = tf.image.random_brightness(image, max_delta=0.5)  # Adjust brightness
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)  # Adjust contrast

    return image

# Setup base model and freeze its layers (this will extract features)
base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False)
base_model.trainable = False

input_size = (200, 200, 3)
inputs = Input(shape=input_size)
X = base_model(inputs,training=False)
X = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
X = BatchNormalization(axis=3)(X)
X = MaxPooling2D((3, 3))(X)

X = Conv2D(128, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)

X = Conv2D(256, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2))(X)

X = Flatten()(X)

dense_1 = Dense(256, activation='relu')(X)
dense_2 = Dense(256, activation='relu')(X)
dense_3 = Dense(128, activation='relu')(dense_2)
dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_3)

# Gender output
output_gender = Dense(1, activation='sigmoid', name='gender_output')(dropout_1)

# Age output
output_age = Dense(1, activation='linear', name='age_output')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_gender, output_age])

model.load_weights("/model_bucket_weights.h5")

def assign_age(age):
    if int(age) <= 12:
        return int(age) * 6 - 3
    elif int(age) == 13:
        return 77
    elif int(age) == 14:
        return 85
    elif int(age) == 15:
        return 95
    elif int(age) == 16:
        return 110
    else:
        return 115  # If age is beyond the specified buckets
    
# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)



if (args.type=='video'):
    # Initialize video capture
    vid_file = input("Enter the filename: ")
    vid_filename, extension = vid_file.split('.')
    video_capture = cv2.VideoCapture(vid_file)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_video.avi", fourcc, fps, (frame_width, frame_height))

    # Dictionary to store age predictions for each tracked face
    tracked_faces = {}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to RGB (if needed)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for face in boxes:
                x, y, w, h = face
                # Extract face region
                face_region = frame_rgb[int(y):int(y + h), int(x):int(x + w)]

                # Super resolution on face region
                enhanced_face_region = rrdn.predict(face_region)
                if(tf.size(enhanced_face_region)):
                    img = preprocess_image(enhanced_face_region)
                    # Predict age
                    preds = model.predict(tf.expand_dims(img, axis=0))
                    age = assign_age(preds[1].reshape(-1))
                    gender = preds[0].reshape(-1)
                    gender_pred = "Male" if gender <= 0.5 else "Female"

                    # Find similar faces
                    similar_face_found = False
                    for tracked_face_box, tracked_age in tracked_faces.items():
                        if np.linalg.norm(np.array(face)[:2] - np.array(tracked_face_box)[:2]) < 20:
                            # Update age prediction for similar face
                            age = tracked_age
                            similar_face_found = True
                            break

                    # If similar face is not found, add the current face to tracked faces
                    if not similar_face_found:
                        tracked_faces[(x, y, w, h)] = age

                    # Draw rectangle around the face
                    # cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), thickness=2)

                    # Add age prediction to the frame
                    cv2.putText(frame, f"Age: {age}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.putText(frame, f"Gender: {gender_pred}", (int(x), int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Write frame to output video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release video capture and video writer
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(list(tracked_faces.items()), columns=['Tuple', 'Value'])
    df.to_csv("output.csv", index=False)

if (args.type=='image'):
    img_file = input("Enter the image filename: ")
    img_filename, extension = img_file.split('.')
    img = cv2.imread(img_file)
    tracked_faces = {}
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is not None:
        for face in boxes:
            x, y, w, h = face
            # Extract face region
            face_region = frame_rgb[int(y):int(y + h), int(x):int(x + w)]
            # Super resolution on face region
            enhanced_face_region = rrdn.predict(face_region)
            if(tf.size(enhanced_face_region)):
                img = preprocess_image(enhanced_face_region)
                # Predict age
                preds = model.predict(tf.expand_dims(img, axis=0))
                age = assign_age(preds[1].reshape(-1))
                gender = preds[0].reshape(-1)
                gender_pred = "Male" if gender <= 0.5 else "Female"

                # Find similar faces
                similar_face_found = False
                for tracked_face_box, tracked_age in tracked_faces.items():
                    if np.linalg.norm(np.array(face)[:2] - np.array(tracked_face_box)[:2]) < 20:
                        # Update age prediction for similar face
                        age = tracked_age
                        similar_face_found = True
                        break

                # If similar face is not found, add the current face to tracked faces
                if not similar_face_found:
                    tracked_faces[(x, y, w, h)] = age

                # Draw rectangle around the face
                # cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), thickness=2)

                # Add age prediction to the frame
                cv2.putText(frame, f"Age: {age}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(frame, f"Gender: {gender_pred}", (int(x), int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    df = pd.DataFrame(list(tracked_faces.items()), columns=['Tuple', 'Value'])
    df.to_csv("output.csv", index=False)

