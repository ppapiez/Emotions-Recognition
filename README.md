# Emotions-Recognition

The aim of this project is to train neural networks to recognize human facial expressions.

Currently, I am building environment needed to conduct research.

Funcionality:
- training neural networks of chosen architecture
- saving and loading of trained network with history of training
- classifying single images using loaded neural network

# Datasets

Prepare such directories:
- Data
  - Kaggle-Emotions
    - fer2013.tar.gz
  - Single images to classify
    - subject01_centerlight.png
    - subject01_glasses.png
    - subject01_happy_small.png
    - ...
  - Yalefaces
    - subject01.centerlight
    - subject01.glasses
    - subject01.happy
    - ...
  
Kaggle-Emotions: 
  - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  - Categories: 'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
  
Yalefaces: 
  - http://vision.ucsd.edu/content/yale-face-database
  - Categories: 'centerlight': 0, 'glasses': 1, 'happy': 2, 'leftlight': 3, 'noglasses': 4, 'normal': 5, 'rightlight': 6,
         'sad': 7, 'sleepy': 8, 'surprised': 9, 'wink': 10
  
# API

Class Manager gives us mentioned funcionality.

We initialize class Manager with args:
- new_model: 
  + 'True', 'False'
  + Choose if you want to create new model or use old one.
- model_architecture: 
  + 'cnn_yf_1', 'cnn_ke_1'
  + Choose which network architecture should be used when creating a new model.
- output_classes: 
  + 'kaggle_classes', 'yalefaces_classes'
  + Set how the output of network should be understood when classifying an image.
- path: 
  + './Models/X'
  + Set path to directory X where trained model is or where new model should be saved.

Methods:
- train_model(args) where args:
  + dataset: 
    - 'Kaggle-Emotions', 'Yalefaces'
  + batch_size
  + epochs
- save_model()
- classify_image(path)
  + Classify an image with a given path. Image should be in proper shape and in .png format.
- plot_history()
  + Show history of loss and accuracy during training.
  
# Models
In this directory you can find two basic CNN architectures:
- cnn_ke_1
- cnn_yf_1

You can find here also directories with trained models. They contain 3 files:
- model.json --- architecture of network
- model.h5 --- weights of network
- history --- history of loss and accuracy during training




