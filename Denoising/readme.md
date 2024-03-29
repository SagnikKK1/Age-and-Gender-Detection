# Age and Gender Detection from Images

## Overview

This project aims to detect the age and gender of individuals from images using computer vision techniques. As part of the preprocessing pipeline, a denoising step is applied to improve the quality of input images. In this project, the BM3D (Block-Matching 3D Filtering) algorithm is employed for denoising face images.

## Motivation

Accurate age and gender detection from images is valuable for various applications, including demographic analysis, targeted advertising, and personalized services. However, image quality can significantly affect the performance of detection algorithms. By applying denoising techniques such as BM3D, we can enhance image quality and improve the accuracy of subsequent age and gender detection.

## Denoising with BM3D

The `denoising` folder contains the implementation of the BM3D algorithm for denoising face images. This step is crucial for preprocessing input images before feeding them into the age and gender detection models.

## Denoising with Autoencoder

In addition to the BM3D algorithm, we explored the utilization of an autoencoder network for denoising face images. Autoencoder networks have shown promising results in image denoising tasks by learning efficient representations of clean image patches. However, due to computational limitations and resource constraints, we were unable to complete the experimentation phase and obtain conclusive results with the autoencoder-based approach.

While the autoencoder network offers potential advantages such as end-to-end learning and adaptability to different noise levels, further investigation and optimization are required to leverage its capabilities effectively. Future iterations of this project may involve revisiting the autoencoder-based denoising approach with improved computational resources and algorithmic enhancements.


## Usage

### Dependencies

- Python 3.x
- NumPy
- OpenCV
- bm3d


### Running the Denoising Algorithm

To denoise face images using BM3D, follow these steps:

1. Place your face images in the `denoising/input_images` directory.
2. Run the `denoising/bm3d_denoise.ipynb` 


## Results

Here are some examples of face images before and after applying BM3D denoising:
BM3d has a PSNR of 23.57


![Original Face Image](Results_images/Noisy_image.png) ![Denoised Face Image](Results_images/bm3d_image.png)

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

