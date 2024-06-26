# Flower Classification Using CNN

This repository contains a Jupyter Notebook that demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) for image classification of different flower types. The notebook includes steps for data loading, model construction, training, and evaluation.

## Dataset

The dataset used consists of images of 16 different flower types. The flower dataset can be found in this repository under the folder `Flower Dataset`.

## Notebook Overview

### Importing Necessary Libraries

The notebook starts by importing essential libraries such as PyTorch, Torchvision, and Matplotlib for deep learning and data visualization.

### Data Loading

- The images are loaded using the `ImageFolder` class from Torchvision.
- The dataset is split into training (80%) and validation (20%) sets.
- Data loaders are created for easy batch processing during training and validation.

### Model Definition

A CNN model (`FlowerCNN`) is defined with the following architecture:
- 6 convolutional layers with ReLU activations
- Max pooling layers
- Fully connected layers

### Parameters

The model is trained using:
- Cross-entropy loss function
- Adam optimizer with a learning rate of 0.001

### Training and Evaluation Functions

- A training function is defined to train the model for a specified number of epochs.
- An evaluation function is defined to calculate the validation loss and accuracy.

### Training the Model

The model is trained for 10 epochs, with the training loss and validation accuracy printed after each epoch.

### Analyzing the Results

- The notebook plots the training losses and validation accuracies over the epochs.
- The final validation accuracy achieved is around 60%.

### Conclusion

The model achieves a validation accuracy of 60%, which is a reasonable performance given the complexity of the task and the dataset.

## How to Use

1. Clone the repository.
2. Ensure you have all necessary dependencies installed.
3. Open the Jupyter Notebook and run the cells sequentially to load the data, define the model, and train it.

## Future Work

Further improvements could include:
- Pretraining a model like ResNet18 on this dataset and comparing its performance with the current model.
- Experimenting with different data augmentation techniques to improve model accuracy.

