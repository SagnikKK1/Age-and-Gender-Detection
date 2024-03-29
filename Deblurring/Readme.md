# MPRNet

## Overview

MPRNet (Motion-Parallax-Retinex Network) is a cutting-edge deep learning-based approach designed for image deblurring tasks. It is particularly tailored for scenarios where motion blur is present, such as images captured in dynamic environments or by devices with limited stabilization capabilities like GoPro cameras. Here we require image deblurring so as to better identify the face of a person if present inside an image. MPRNet utilizes a combination of advanced techniques, including motion analysis, parallax estimation, and Retinex-based processing, to effectively enhance image sharpness and clarity.

## Methodology

MPRNet operates through a multi-stage process to tackle the challenges posed by motion blur:

1. **Motion Analysis:** MPRNet begins by analyzing the motion characteristics present in the input images. It estimates motion parameters and blur kernels to understand the underlying motion patterns and guide the deblurring process effectively.

2. **Parallax Estimation:** Leveraging the parallax information inherent in the scene, MPRNet identifies and rectifies geometric distortions caused by motion blur. By aligning image features accurately, it enhances the overall clarity and visual quality of the deblurred images.

3. **Retinex-Based Processing:** MPRNet incorporates Retinex-based processing techniques to further improve image contrast and luminance. This step enhances the visibility of subtle details and textures that may have been obscured by motion blur, resulting in sharper and more visually appealing images.

## Requirements

To use MPRNet for image deblurring tasks, the following dependencies are required:

- Python 3.x
- PyTorch
- NumPy
- OpenCV

