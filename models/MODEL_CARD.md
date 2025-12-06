
# Flower Recognition Models for Scent Dispensing

## Model Files
- vgg16_flower_classifier_final.keras
- resnet50_flower_classifier_final.keras  
- mobilenet_flower_classifier_final.keras

## Model Specifications
- Input Shape: (224, 224, 3)
- Number of Classes: 102
- Output: Softmax probabilities for each flower class

## Performance Summary
- VGG16 Accuracy: 43.32%
- ResNet50 Accuracy: 2.24%
- MobileNetV2 Accuracy: 73.51%

## Usage
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('vgg16_flower_classifier_final.keras')

# Prepare image
image = preprocess_image(your_image)  # Resize to (224, 224), normalize to [0,1]

# Predict
predictions = model.predict(image)
flower_class = np.argmax(predictions[0])
confidence = predictions[0][flower_class]
```

## Training Details
- Dataset: Oxford 102 Flowers
- Training Samples: 1020
- Test Samples: 6149
- Epochs: 10
- Batch Size: 32
- Optimizer: Adam
- Data Augmentation: Rotation, Flipping, Brightness/Contrast adjustment
