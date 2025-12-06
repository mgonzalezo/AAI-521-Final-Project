# Fast Flower Recognition for Scent Dispensing System

## AAI-521 Final Project - Group 4
**Date:** December 2025

> **Note:** This project applies topics and CNN models studied in the AAI-521 Computer Vision and Image Processing course, including Convolutional Neural Networks (CNNs), Transfer Learning, Data Augmentation, Video Processing, and Deep Learning optimization techniques. The implementation leverages concepts and methodologies covered throughout the course curriculum.

---

## Project Overview

### Title
**Fast Object Recognition of Flower Type in Video Scene for Further Action to Dispense the Perfume/Scent of the Identified Flower**

### Objective
While special effects in vision and sound have been optimized to create immersive movie-watching experiences, the sense of smell has not been addressed. This project demonstrates the feasibility of identifying flower types in real-time video scenes to enable synchronized scent/perfume dispensing in movie theaters and home theater environments, thereby enhancing the total enjoyment experience.

### Key Goals
- Develop a computer vision system for real-time flower classification
- Achieve minimal time lag (< 130ms per frame) for synchronized scent dispensing
- Demonstrate proof-of-concept for scene-based scent delivery systems
- Compare multiple deep learning architectures for accuracy and speed

### Scope
- **In Scope:** Computer vision model for flower detection and recognition in video
- **Out of Scope:** Hardware implementation of perfume dispensing mechanism
- **Use Case:** Flower identification (extensible to gardens, waterfalls, forests, beaches)

---

## Dataset Information

### Primary Dataset: Oxford 102 Flowers
- **Source:** [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Classes:** 102 flower categories
- **Samples per Class:** 40-258 images
- **Total Images:** ~8,000+ images
- **Characteristics:** Large scale, pose, and light variations
- **Train/Test Split:** 70% Training / 30% Testing (as specified)

### Dataset Access
The project uses TensorFlow Datasets for easy access:
```python
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```

### Data Preprocessing
Following techniques from the reference paper (Tian et al., 2019):
- Image resizing to 224×224 pixels
- Normalization to [0, 1] range
- Data augmentation:
  - Random horizontal flipping
  - Random rotation (90°, 180°, 270°, 360°)
  - Random brightness adjustment
  - Random contrast adjustment
  - Random saturation adjustment

---

## Project Structure

```
AAI-521-Final-Project/
├── data/                          # Dataset storage (auto-downloaded)
├── models/                        # Trained models
│   ├── vgg16_flower_classifier_final.keras
│   ├── resnet50_flower_classifier_final.keras
│   ├── mobilenet_flower_classifier_final.keras
│   └── MODEL_CARD.md
├── results/                       # Visualizations and reports
│   ├── sample_images.png
│   ├── class_distribution.png
│   ├── training_history_comparison.png
│   ├── confusion_matrix_vgg16.png
│   ├── per_class_accuracy_vgg16.png
│   ├── inference_speed_comparison.png
│   ├── model_comparison_summary.png
│   ├── classification_report_vgg16.txt
│   └── final_project_report.txt
├── flower_recognition_scent_dispenser.ipynb  # Main notebook
└── README.md                      # This file
```

---

## Implementation Details

### 1. Project Selection & Setup
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn
- **Hardware:** Compatible with CPU and GPU (CUDA support optional)
- **Configuration:**
  - Image size: 224×224 pixels
  - Batch size: 32
  - Epochs: 50 (with early stopping)
  - Learning rate: 0.001
  - Optimizer: Adam

### 2. Exploratory Data Analysis (EDA)
- Sample image visualization
- Class distribution analysis
- Image property statistics (dimensions, aspect ratios)
- Data augmentation visualization

### 3. Modeling Methods

Three deep learning architectures were implemented and compared:

#### VGG16 (Primary Model - Based on Reference Paper)
- **Base:** Pre-trained on ImageNet
- **Architecture:** VGG16 feature extractor + custom classification head
- **Layers:** GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Dense(102)
- **Rationale:** Used in the reference paper (Tian et al., 2019) with proven results

#### ResNet50 (Deep Architecture)
- **Base:** Pre-trained on ImageNet
- **Architecture:** ResNet50 feature extractor + custom classification head
- **Layers:** GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout → Dense(102)
- **Rationale:** Deeper network for potentially better feature extraction

#### MobileNetV2 (Fast Inference)
- **Base:** Pre-trained on ImageNet
- **Architecture:** MobileNetV2 feature extractor + custom classification head
- **Layers:** GlobalAveragePooling2D → Dense(256) → BatchNorm → Dropout → Dense(102)
- **Rationale:** Optimized for speed, critical for real-time scent dispensing

### 4. Validation and Performance Metrics

#### Evaluation Metrics
- **Accuracy:** Primary metric for classification performance
- **Top-5 Accuracy:** Allows for multiple scent options
- **Confusion Matrix:** Per-class performance analysis
- **Precision, Recall, F1-Score:** Detailed classification metrics
- **Inference Time:** Critical for real-time processing (target: < 130ms)

#### Training Callbacks
- **EarlyStopping:** Prevents overfitting (patience=10)
- **ModelCheckpoint:** Saves best model
- **ReduceLROnPlateau:** Adaptive learning rate

### 5. Results and Findings

Results will be generated when you run the notebook. Expected performance based on similar implementations:
- **Accuracy:** 70-90% (depending on model)
- **Inference Speed:** Varies by model (VGG16 typically ~50-100ms, MobileNetV2 faster)
- **Real-time Capability:** Achievable with optimized models

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone or Navigate to Project Directory
```bash
cd /home/margonza/Documents/Marco/Master/AAI-521-IN2/Final_project/AAI-521-Final-Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Using conda
conda create -n flower-recognition python=3.8
conda activate flower-recognition
```

### Step 3: Install Required Packages
```bash
pip install tensorflow tensorflow-datasets tensorflow-hub
pip install opencv-python numpy pandas matplotlib seaborn
pip install scikit-learn pillow ipython jupyter
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook flower_recognition_scent_dispenser.ipynb
```
Then execute cells sequentially from top to bottom.

### Option 2: JupyterLab
```bash
jupyter lab
```
Navigate to `flower_recognition_scent_dispenser.ipynb` and run cells.

### Option 3: VS Code
- Open the project folder in VS Code
- Install the Jupyter extension
- Open `flower_recognition_scent_dispenser.ipynb`
- Click "Run All" or execute cells individually

---

## Step-by-Step Execution Guide

### Phase 1: Setup and Data Loading (Cells 1-4)
1. **Import Libraries:** Load all required dependencies
2. **Configure Project:** Set paths and hyperparameters
3. **Load Dataset:** Download Oxford 102 Flowers via TensorFlow Datasets
4. **Verify Setup:** Check versions and GPU availability

**Expected Time:** 2-5 minutes (depending on download speed)

### Phase 2: EDA and Preprocessing (Cells 5-10)
1. **Visualize Samples:** Display random flower images
2. **Analyze Distribution:** Check class balance
3. **Image Properties:** Analyze dimensions and aspect ratios
4. **Create Pipelines:** Set up preprocessing and augmentation
5. **Visualize Augmentation:** Confirm transformations work correctly

**Expected Time:** 5-10 minutes

### Phase 3: Model Creation and Compilation (Cells 11-14)
1. **Build VGG16 Model:** Create primary architecture
2. **Build Alternative Models:** Create ResNet50 and MobileNetV2
3. **Compile Models:** Set optimizer, loss, and metrics
4. **Configure Callbacks:** Set up training callbacks

**Expected Time:** 1-2 minutes

### Phase 4: Model Training (Cells 15-17)
1. **Train VGG16:** Primary model training (~30-60 minutes)
2. **Train ResNet50:** Alternative model training (~30-60 minutes)
3. **Train MobileNetV2:** Fast model training (~20-40 minutes)

**Expected Time:** 1.5-3 hours total
**Note:** You can train just one model initially to save time

### Phase 5: Evaluation and Analysis (Cells 18-25)
1. **Plot Training History:** Visualize accuracy and loss curves
2. **Evaluate Models:** Test set performance
3. **Generate Predictions:** Create prediction arrays
4. **Confusion Matrix:** Analyze per-class performance
5. **Classification Report:** Detailed metrics
6. **Inference Speed Test:** Measure processing time

**Expected Time:** 10-15 minutes

### Phase 6: Results and Reporting (Cells 26-29)
1. **Model Comparison:** Compare all three architectures
2. **Sample Predictions:** Visualize model outputs
3. **Video Demo:** Simulate real-time detection
4. **Final Report:** Generate comprehensive findings
5. **Save Models:** Export trained models

**Expected Time:** 5-10 minutes

---

## Usage Examples

### Loading a Trained Model
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('models/vgg16_flower_classifier_final.keras')

# Load and preprocess image
img = Image.open('path/to/flower_image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_batch)
top_class = np.argmax(predictions[0])
confidence = predictions[0][top_class]

print(f"Predicted Class: {top_class}")
print(f"Confidence: {confidence*100:.2f}%")
```

### Real-Time Video Processing
```python
import cv2

# Open video
cap = cv2.VideoCapture('flower_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype('float32') / 255.0
    batched = np.expand_dims(normalized, axis=0)

    # Predict
    predictions = model.predict(batched, verbose=0)
    top_class = np.argmax(predictions[0])
    confidence = predictions[0][top_class]

    # Display result
    if confidence > 0.7:  # Threshold for scent trigger
        cv2.putText(frame, f"Flower: {top_class} ({confidence*100:.1f}%)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"TRIGGER SCENT: Class {top_class}")

    cv2.imshow('Flower Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Alignment with Course Topics

This project applies key concepts and techniques from the AAI-521 Computer Vision and Image Processing course:

### Core Course Topics Applied
- **Convolutional Neural Networks (CNNs):** Feature extraction and classification using deep learning architectures
- **Transfer Learning:** Leveraging pre-trained ImageNet models (VGG16, ResNet50, MobileNetV2)
- **Data Augmentation:** Rotation, flipping, brightness/contrast adjustment to improve model robustness
- **Video Processing:** Frame-by-frame analysis for real-time flower detection
- **Model Evaluation:** Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrices

### Technical Implementation
- **Image Preprocessing:** Normalization, resizing, and batch processing pipelines
- **OpenCV Integration:** Video handling and image transformations
- **Training Optimization:** Adam optimizer, learning rate scheduling, early stopping
- **Performance Analysis:** Inference speed measurement for real-time requirements
- **Visualization:** Matplotlib and Seaborn for EDA and results presentation

---

## Reference Paper Implementation

This project is inspired by:
**Tian, M., Chen, H., & Wang, Q. (2019).** "Detection and Recognition of Flower Image Based on SSD network in Video Stream." *Journal of Physics: Conference Series*, 1237, 032045.

### Key Implementations from Paper:
1. **VGG16 Architecture:** Used as primary feature extractor
2. **Data Augmentation:** Rotation (90°, 180°, 270°), horizontal flipping
3. **Performance Target:** < 130ms inference time per image
4. **Evaluation Standards:** Pascal VOC-style metrics
5. **Application Context:** Real-time video stream processing

### Differences from Paper:
- **Dataset:** Oxford 102 Flowers (102 classes) vs. paper's 19 classes
- **Architecture:** Added ResNet50 and MobileNetV2 for comparison
- **Framework:** TensorFlow/Keras vs. paper's implementation
- **Objective:** Scent dispensing vs. general flower recognition

---

## Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow Installation Issues
```bash
# If GPU support needed
pip install tensorflow[and-cuda]

# If only CPU needed
pip install tensorflow-cpu
```

#### 2. Dataset Download Fails
```bash
# Manually set download directory
export TFDS_DATA_DIR=/path/to/data
```

#### 3. Out of Memory Errors
```python
# Reduce batch size in configuration
BATCH_SIZE = 16  # or even 8
```

#### 4. Slow Training
```python
# Use smaller number of epochs for testing
EPOCHS = 10  # Instead of 50

# Or train only one model first
# Comment out ResNet50 and MobileNetV2 training cells
```

#### 5. GPU Not Detected
```python
# Verify GPU availability
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))

# If empty, check CUDA installation
```

---

## Acknowledgments

- **Reference Paper:** Tian et al. (2019) for SSD-based flower detection methodology
- **Dataset:** Oxford Visual Geometry Group for the Oxford 102 Flowers dataset
- **TensorFlow Team:** For TFDS and pre-trained models
- **Course Instructors:** For guidance on deep learning and computer vision techniques throughout AAI-521

---

## Team Members

**Group 4 - AAI-521 Computer Vision and Image Processing**

### Team Members (Full Names):
- Vijay Agarwal
- Marco Antonio Gonzalez
- Ritesh Jain

### Team Leader/Representative:
Marco Antonio Gonzalez

---

**December 2025**

For questions or discussions about this project, please refer to course communication channels.
