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

