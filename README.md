# Age and Gender Detection System

## Overview
This project aims to develop an age and gender detection system capable of accurately determining age and gender from human face images and real-time video streams from a mobile camera. The solution is capable of detecting the age and gender of at max 5 people in the image or video.

## Project Structure

The project repository contains the following folders and files:

1. **Age_Prediction_1**
- **checkpoint**: Contains saved checkpoints for the trained models.
- **img**: Contains sample images for testing the prediction script.
- **video**: Contains sample videos for testing the prediction script.
- **prediction**: Output folder for storing prediction results.
- **predict.py**: Script for making predictions on images or videos.
- **predict_1.py**: Script for making predictions on images or videos with our custom trained model

2. **Age_Prediction_2**
- **predict.py**

3. **C++**
- **prediction** - Contained sample results
- **age_and_gender_prediction.cpp** - Script for making predictions on images or videos.

## Model Architecture and Training

### Model Architecture
The model architecture consists of a custom neural network with skip CNN connections, using EfficientNetB7 as the backbone structure. Here's a summary of the model:

- Base model: EfficientNetB7 (pretrained on ImageNet, frozen during training)
- Custom layers for age and gender prediction
- Skip CNN connections for feature extraction
- Final output layers for age and gender prediction

### Training
- Trained on a Tesla T4 GPU on Google Colab.
- Training time: Approximately 40 minutes.
- Training metrics:
  - Mean Absolute Error (MAE) for age prediction: ~6.6 on the test dataset.
  - Accuracy for gender prediction: ~88% on the test dataset.

### Fine-Tuning
- The last five layers of the EfficientNetB7 were unfreezed and fine-tuned for another 10 epochs to improve performance.

## Predict Script (predict.py)

### Requirements
- GPU access is recommended for faster prediction.
- Required libraries: PyTorch, TensorFlow, OpenCV, NumPy, pandas, tqdm, matplotlib, PIL.

### Usage
- **Arguments**:
  - `--type`: Type of input (`image` or `video`).
- **Predicting on Images**:
  - Provide the image filename when prompted.
  - Predictions will be saved in the `prediction/{image_filename}` directory.
- **Predicting on Videos**:
  - Provide the video filename when prompted.
  - Predictions will be saved in the `prediction/{video_filename}` directory.
  - Output video with bounding boxes and person IDs will be saved if enabled.

### Prediction Process
- Load the input image or video frame by frame.
- Use a pretrained Faster R-CNN model to detect people's faces.
- For each detected face:
  - Extract the face region.
  - Preprocess the image (resize, normalize).
  - Pass it through the trained age and gender detection model.
  - Draw bounding boxes, predicted age, and gender on the frame.
- Save the results in CSV format and optionally save frame images or output video.

## Additional Notes
- Ensure all required libraries and pretrained models are available.
- Adjust the paths for the models and checkpoints as necessary.
- For optimal performance, run the prediction script on a machine with GPU support.

---

Feel free to customize this README further based on additional details or requirements specific to your project.
