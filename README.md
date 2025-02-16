# Pneumonia Detection

## Overview
This project is a **Pneumonia Detection** web app built with **Streamlit**. It uses a deep learning model trained on chest X-ray images to classify whether a given X-ray indicates pneumonia or not.

## Features
- Upload a chest X-ray image.
- Get a classification result (Pneumonia or Normal) along with a confidence score.
- Background customization using a selected image.

## Tech Stack
- **Python**
- **Streamlit** (for UI)
- **TensorFlow/Keras** (for deep learning model)
- **Pillow (PIL)** (for image processing)
- **NumPy** (for array operations)

## Installation
### Clone the Repository
```bash
git clone https://github.com/rohanbhoyar234/PneumoniaDetection.git
cd PneumoniaDetection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run main.py
```

## Project Structure
```
PneumoniaDetection/
│── main.py                 # Streamlit app main file
│── util.py                 # Helper functions for background and classification
│── pneumonia_classifier.h5 # Trained model file
│── labels.txt              # Class labels
│── bg2.jpeg                # Background image
│── output_images/          # Folder containing sample output images
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation
```

## Usage
1. Run the app using `streamlit run main.py`.
2. Upload a chest X-ray image (JPEG, JPG, or PNG).
3. The model will classify the image as **Pneumonia** or **Normal** with a confidence score.


## Model Details
- The model was trained using **CNNs (Convolutional Neural Networks)**.
- Input image is resized to **224x224 pixels** before classification.
- Uses a **custom threshold of 0.95** to determine pneumonia classification.

## Future Improvements
- Improve model accuracy with more training data.
- Deploy the model on a cloud platform for wider accessibility.
- Enhance UI with additional diagnostic features.



