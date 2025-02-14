# Age and Gender Detection System

## Overview
This project aims to develop an age and gender detection system capable of accurately determining age and gender from human face images and real-time video streams from a mobile camera. The solution is capable of detecting the age and gender of at max 5 people in the image or video.


## Project Structure

The project repository is organized as follows:

### Age_Prediction_1
- checkpoint: Contains saved checkpoints for the trained models.
- img: Contains sample images for testing the prediction script.
- video: Contains sample videos for testing the prediction script.
- prediction: Output folder for storing prediction results.
- predict.py: Script for making predictions on images or videos.
- predict_1.py: Script for making predictions on images or videos with our custom trained model.

### Age_Prediction_2
- predict.py: Script for making predictions on images or videos.

### C++
- prediction: Contains sample results.
- age_and_gender_prediction.cpp: C++ script for making predictions on images or videos.
- age_deploy.prototxt: Age prediction model architecture file.
- gender_deploy.prototxt: Gender prediction model architecture file.
- opencv_face_detector.pbtxt: Face detection model configuration file.
- opencv_face_detector_uint8.pb: Face detection model weights file.

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

## Age_Prediction_1
### Predict Script (predict.py)

#### Requirements
- GPU access is recommended for faster prediction.
- Required libraries: PyTorch, TensorFlow, OpenCV, NumPy, pandas, tqdm, matplotlib, PIL.

#### Usage
- **Arguments**:
  - `--type`: Type of input (`image` or `video`).
- **Predicting on Images**:
  - Provide the image filename when prompted.
  - Predictions will be saved in the `prediction/{image_filename}` directory.
- **Predicting on Videos**:
  - Provide the video filename when prompted.
  - Predictions will be saved in the `prediction/{video_filename}` directory.
  - Output video with bounding boxes and person IDs will be saved if enabled.

#### Prediction Process
- Load the input image or video frame by frame.
- Use a pretrained Faster R-CNN model to detect people's faces.
- For each detected face:
  - Extract the face region.
  - Preprocess the image (resize, normalize).
  - Pass it through the trained age and gender detection model.
  - Draw bounding boxes, predicted age, and gender on the frame.
- Save the results in CSV format and optionally save frame images or output video.

### **predict_1.py** script
- predict_1.py is an alternative prediction script that employs a custom neural network model for age and gender detection

### Usage 
- Make sure to download all the checkkpoint paths from [drive](https://drive.google.com/drive/folders/1_4m8BhuVMxbYW88SoS8KEo1hsdzsmecJ?usp=drive_link) and place it under checkpoint folder under Age_Prediction_1
- Make sure to place the required video and images under the video and image folder respectively under Age_Prediction_1
- When prompted to enter the file name, make sure to enter the file extension also. e.g. If the filename is "test.mp4" then enter "test.mp4"
- Download all the requirements using `pip install -r requirements.txt`
- Finally run the following code for prediction on video
- `python predict.py -- video`
- To obtain prediction on image run the following command
- `python predict.py -- image`
- A example ipynb has been given for reference

Similary predict_1.py can be run


## Age_Prediction_2
- The prediction process is facilitated by a custom neural network model for age and gender detection, along with a pre-trained model for super-resolution enhancement. Additionally, face detection is carried out using a pre-trained MTCNN (Multi-Task Cascaded Convolutional Neural Network) model.

### Custom Model for Age and Gender Detection
- Architecture
The custom neural network model is designed for age and gender detection from facial images. For age prediction, the age groups are bucketed into ranges such as (1-6), (7-12), etc., to facilitate easier classification, particularly in video prediction scenarios. The age prediction model outputs the mean age of each bucket as the final predicted age.
- Training
The model is trained on a dataset with annotated age and gender labels. During training, age labels are grouped into buckets, simplifying the classification task. The training process aims to optimize the model parameters to accurately predict both age and gender from facial images.
- Bucketed Age Groups
Age groups are divided into buckets to ease classification, especially in scenarios with multiple faces in a video stream. Each bucket represents a range of ages, such as (1-6), (7-12), etc.
The final predicted age for each face is the mean age of the corresponding bucket.

### Super Resolution Model
- Library: The project utilizes the image super-resolution library available at idealo/image-super-resolution.
- Model: The RRDN (Residual in Residual Dense Network) model is used for super-resolution enhancement.
- Loading: The RRDN model is initialized with pre-trained weights for image enhancement.

### Face Detection Model
- Library: Face detection is performed using the MTCNN (Multi-Task Cascaded Convolutional Neural Network) model.
- Initialization: MTCNN is initialized from the facenet_pytorch library.
- Device: The model is run on GPU if available; otherwise, it falls back to CPU.

### Prediction Process
#### Video Prediction
- The script reads frames from the input video (vid_1.mp4).
- Faces are detected using the MTCNN model.
- Super-resolution is applied to each detected face region using the RRDN model.
- Age and gender predictions are made for each face region using the custom model.
- Predicted age and gender are annotated on the frames, and the processed frames are written to the output video (output_1.avi).

#### Image Prediction
- The script reads the input image provided by the user.
- Faces are detected using the MTCNN model.
- Super-resolution is applied to each detected face region using the RRDN model.
- Age and gender predictions are made for each face region using the custom model.
- Predicted age and gender are annotated on the image, and the processed image is saved with the suffix _output.

### Usage
- Download all the requirements using `pip install -r requirements.txt`
- Follow all the steps in the ipynb to get predictions
- Make sure to download `model_bucket_weights.h5` from the [here](https://drive.google.com/drive/folders/1_4m8BhuVMxbYW88SoS8KEo1hsdzsmecJ?usp=drive_link) and place it in the working directory (in Age_Prediction_2 folder)
- Carefully enter the input and output file path. Save output fiile with `.avi` extension

## Age and Gender prediction using C++
The C++ program detects faces in images or videos using OpenCV's deep neural network module (dnn). It then predicts the age and gender of each detected face using pre-trained models. The age and gender prediction models are based on Convolutional Neural Networks (CNNs) implemented in Caffe.

### Prerequisites
Make sure you have the following installed on your system:

- OpenCV 4.x or higher
- CMake (for building the C++ code)

The program supports two modes of operation:

- Image Processing: Process a single image.
- Video Processing: Process a video file.

### Model Files
The program requires the following pre-trained model files:

- age_deploy.prototxt and age_net.caffemodel: Age prediction model.
- gender_deploy.prototxt and gender_net.caffemodel: Gender prediction model.
- opencv_face_detector_uint8.pb and opencv_face_detector.pbtxt: Face detection model.
Ensure these model files are either placed in the same directory as the executable or specify their paths when prompted.

### Output
For each detected face, the program outputs the predicted gender and age group in the format Gender: <gender>, Age: <age group>.

Additionally, the program draws bounding boxes around detected faces and overlays the predicted gender and age information on the image or video frames.

## Result
Our C++ script performs best on video prediction and outputs a range of age. The mean age of the bucket may be taken for consideration of MAE.

All our sample results are available under `Prediction_Results` folder [here](https://drive.google.com/drive/folders/1_4m8BhuVMxbYW88SoS8KEo1hsdzsmecJ?usp=drive_link)

Our results for UTKFACE which can be seen in utk_train.ipynb are:
- Test dataset
mae = 6.5,
 mse = 82.9,
gender prediction accuracy = 89%

- Validation dataset
mae = 6.5,
 mse = 83.3,
Gender prediction accuracy = 87.4 % 

- Test dataset 
mae = 4.59,
 mse = 37.9,
gender prediction accuracy = 92.4%

## Future Scopes
- Currently, we have employed the use of a non-ML algorithm known as bm3d for image denoising. We also used autoencoder netwroks for our deblurring task. We aim to integrate Machine Learning models specifically developed for deblurring such as KBnet for denoising images. Also currently we have not integrated the denoised images into the prediction model. We aim to complete that in the future.
- We also aim to integrate Deraining and Deblurring algorithms into the prediction model to furthur improve the range of quality of images that can be used.

  
## Additional Notes
- Ensure all required libraries and pretrained models are available.
- Adjust the paths for the models and checkpoints as necessary.
- For optimal performance, run the prediction script on a machine with GPU support.


### Sources
- https://github.com/rdev12/BOSCH-Age-and-Gender-Prediction


---

