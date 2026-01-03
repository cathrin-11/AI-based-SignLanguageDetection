# AI-Based Sign Language Recognition System

This project recognizes American Sign Language (ASL) alphabets (A–Z) in real time using Deep Learning and Computer Vision.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy, Pandas
- Scikit-learn

## Dataset
- Sign MNIST Dataset
- 28×28 grayscale images
- 26 classes (A–Z)

## How to Run
- python cam.py
- python data_preparation.py
- python train.py
- python real_time_detection.py


## Model
- CNN with Conv2D and MaxPooling layers
- Input shape: (28, 28, 1)
- Output: 26 classes

## Results
- Training accuracy ≈ 99%
- Real-time alphabet prediction using webcam

## Applications
- Communication aid for hearing-impaired people
- Human–computer interaction
