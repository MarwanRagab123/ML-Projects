# Hand Gesture Recognition System

## Project Overview
This project implements a deep learning-based hand gesture recognition system using the Leap Gesture Recognition dataset. The system can classify 9 different hand gestures using a Convolutional Neural Network (CNN) architecture.

## Features
- Real-time hand gesture recognition
- Support for 9 different hand gestures
- High accuracy classification
- Data augmentation for better generalization
- Comprehensive model evaluation and visualization

## Gesture Classes
The system recognizes the following hand gestures:
1. Palm (01_palm)
2. L shape (02_l)
3. Fist (03_fist)
4. Moved Fist (04_fist_moved)
5. Thumb (05_thumb)
6. Index (06_index)
7. OK sign (07_ok)
8. Moved Palm (08_palm_moved)
9. C shape (09_c)

## Technical Details

### Model Architecture
The system uses a CNN with the following structure:
- Input Layer: 64x64 grayscale images
- 3 Convolutional Blocks:
  - Block 1: 32 filters, 3x3 kernel
  - Block 2: 64 filters, 3x3 kernel
  - Block 3: 128 filters, 3x3 kernel
- Each block includes:
  - Conv2D layer
  - BatchNormalization
  - MaxPooling2D
  - Dropout
- Dense Layers:
  - 256 neurons with ReLU activation
  - Output layer with softmax activation

### Data Processing
- Image resizing to 64x64 pixels
- Grayscale conversion
- Normalization (0-1 range)
- Data augmentation:
  - Rotation (±20 degrees)
  - Width/Height shift (±20%)
  - Horizontal flip

### Training Parameters
- Batch size: 32
- Epochs: 50
- Optimizer: Adam (learning rate = 0.001)
- Loss function: Categorical Crossentropy
- Early stopping with patience = 10

## Requirements
```
tensorflow>=2.0.0
numpy>=1.19.2
pandas>=1.1.3
opencv-python>=4.4.0
scikit-learn>=0.23.2
matplotlib>=3.3.2
seaborn>=0.11.0
```

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Download the Leap Gesture Recognition dataset
2. Update the data path in `hand_gesture_recognition.py`
3. Run the script:
```bash
python hand_gesture_recognition.py
```

## Output Files
The system generates the following files:
- `best_model.h5`: Best model weights during training
- `final_model.h5`: Final trained model
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training accuracy and loss plots

## Model Evaluation
The system provides:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Training history plots
- Model accuracy metrics

## Future Improvements
1. Real-time prediction using webcam
2. Support for more hand gestures
3. Mobile deployment
4. Transfer learning implementation
5. Multi-hand gesture recognition
