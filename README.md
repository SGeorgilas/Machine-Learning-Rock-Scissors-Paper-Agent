# Machine-Learning-Rock-Scissors-Paper-Agent




## Project Description

The Rock-Scissors-Paper Agent is designed to create an intelligent agent that learns to play the game Rock-Paper-Scissors. The agent interprets images corresponding to 0: Rock, 1: Scissors, or 2: Paper and makes decisions based on the winning symbol.

## How It Works

Our agent will bet 1€ against a "Random Agent" for a total of N rounds. If our agent wins, it receives 2€; in a tie, it gets 1€ back; otherwise, it loses 1€. The Random Agent always plays first, selecting a random image from the test set of images (representing Rock, Scissors, or Paper). The Random Agent may apply transformations like vertical flip, horizontal flip, and random white noise to the selected image.

## Dataset
You can find the dataset on https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

### Random Agent Operations:

1. Match moves to (0, 1, 2) for (Rock, Scissors, Paper).
2. Select a random image corresponding to a move (0, 1, 2).
3. Preprocess the image:
   a. Apply Vertical Flip with probability p1.
   b. Apply Horizontal Flip with probability p2.
   c. Add white noise with mean 0 and standard deviation 5% of the max pixel value.

### Our Agent Operations:

1. Get a CNN model trained on the training set.
2. Get the augmented picture from the Random Agent and choose the move that beats it.


## Dependencies

The project relies on the following Python libraries:

- [OpenCV (cv2)](https://pypi.org/project/opencv-python/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)

```python
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
```

## Project Structure

The project is organized into the following directories in Google Colab:

- `Rock`: Contains images corresponding to the Rock move.
- `Scissors`: Contains images corresponding to the Scissors move.
- `Paper`: Contains images corresponding to the Paper move.

```python
# Folder paths in Google Colab
rock_folder = '/content/Rock'
scissors_folder = '/content/Scissors'
paper_folder = '/content/Paper'

# Move mapping dictionary
move_mapping = {'Rock': 0, 'Scissors': 1, 'Paper': 2}
```

## Image Loading and Preprocessing

The following function, `load_and_preprocess_image`, is used to load, resize, and normalize images for further processing in the project.

```python
# Function for loading, resizing, and normalizing an image
def load_and_preprocess_image(image_path, move_mapping):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (100, 100)) / 255.0  # Normalize to [0, 1] and resize
    move = os.path.basename(os.path.dirname(image_path))
    move_value = move_mapping.get(move, None)
    return img, move_value
```
### Usage Example
The function is then used to collect images and their corresponding labels for the Rock, Scissors, and Paper moves:
```python

# Collecting images and labels
all_images = []
all_labels = []

for img_path in os.listdir(rock_folder):
    img_path = os.path.join(rock_folder, img_path)
    img_array, move_value = load_and_preprocess_image(img_path, move_mapping)
    all_images.append(img_array)
    all_labels.append(move_value)

# Similar loops for Scissors and Paper folders
# ...

# Converting to numpy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)
```

## RandomAgent Class

The `RandomAgent` class represents an intelligent agent that uses a convolutional neural network (CNN) to make decisions in a Rock-Paper-Scissors game.

### Initialization

```python
# Example initialization
random_agent = RandomAgent(data_folder='/content/data_folder', model_folder='/content/model_folder')
```
The RandomAgent class is initialized with the paths to the data folder (data_folder) containing Rock, Scissors, and Paper images, and the model folder (model_folder) where the trained CNN model will be saved.

##Methods
###load_and_preprocess_image
This method loads and preprocesses an image:

```python
# Example usage
image_path = '/content/Rock/rock_image.jpg'
img = random_agent.load_and_preprocess_image(image_path)
```

###build_and_train_model
This method builds and trains the CNN model using images from the specified data folder:

```python
# Example usage
model, train_images, test_images, train_labels, test_labels = random_agent.build_and_train_model()
```

###choose_random_image
This method randomly chooses an image, optionally from a provided test set:

```python
# Example usage
random_image, move_name = random_agent.choose_random_image(p1=0.5, p2=0.5, noise_std=0.05, test_set_images=test_images, test_labels=test_labels)
```

###augment_image
This method augments an image by applying random flips and adding noise:

```python
# Example usage
augmented_image = random_agent.augment_image(img, p1=0.5, p2=0.5, noise_std=0.05)
```
