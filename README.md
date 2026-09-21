# Image-colorization

Image Colorization is a deep learning application that converts grayscale images into color images using a Vision Transformer (ViT) model and LAB color space.

## Features

* Grayscale Image Upload
* Automatic Image Colorization
* Vision Transformer-based Model
* LAB Color Space Processing
* PSNR and SSIM Evaluation
* Original Image Comparison
* Colorized Image Download

## Tech Stack

* Python
* PyTorch
* Vision Transformer (ViT)
* OpenCV
* NumPy
* scikit-image
* Streamlit

## Project Workflow

1. User uploads a grayscale image.
2. The image is resized to 224×224.
3. The image is converted from RGB to LAB color space.
4. The L channel is provided as input to the trained ViT model.
5. The model predicts the A and B color channels.
6. The predicted channels are combined with the L channel.
7. The LAB image is converted back to RGB.
8. PSNR and SSIM are calculated to evaluate the output.

## Evaluation

The application uses:

* **PSNR** to measure image reconstruction quality.
* **SSIM** to measure structural similarity between images.

The application also displays an SSIM-based accuracy percentage.

## Project Structure

```text
image-colorization/
│
├── app.py
├── model.py
├── vit_best_model.pth
├── requirements.txt
└── README.md
```


streamlit run app.py
```
