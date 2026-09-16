import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from variables import IMAGE_SIZE, LABELS
import matplotlib.pyplot as plt

# Load the data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Ensure input shape is correct (add channel dimension if needed)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

# Convert labels to categorical format if they're not already
if len(y_train.shape) == 1:
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, num_classes=len(LABELS))

# Normalize the data
X_train = X_train / 255.0

# Create data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Don't flip horizontally as it might confuse similar letters
    fill_mode='nearest'
)

def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(LABELS), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# K-fold Cross-validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store metrics
fold_histories = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    print(f'\nTraining Fold {fold + 1}/{n_splits}')
    
    # Split data
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]
    
    # Create and train model
    model = create_model()
    
    history = model.fit(
        datagen.flow(X_train_fold, y_train_fold, batch_size=32),
        validation_data=(X_val_fold, y_val_fold),
        epochs=50,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Evaluate model
    scores = model.evaluate(X_val_fold, y_val_fold)
    print(f'Fold {fold + 1} - Validation Accuracy: {scores[1]*100:.2f}%')
    
    fold_histories.append(history.history)
    fold_scores.append(scores[1])

# Save the final model
model.save('model.h5')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot average accuracy
plt.subplot(1, 2, 1)
mean_acc = np.mean([hist['accuracy'] for hist in fold_histories], axis=0)
mean_val_acc = np.mean([hist['val_accuracy'] for hist in fold_histories], axis=0)
plt.plot(mean_acc)
plt.plot(mean_val_acc)
plt.title('Model Accuracy (Average across folds)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

# Plot average loss
plt.subplot(1, 2, 2)
mean_loss = np.mean([hist['loss'] for hist in fold_histories], axis=0)
mean_val_loss = np.mean([hist['val_loss'] for hist in fold_histories], axis=0)
plt.plot(mean_loss)
plt.plot(mean_val_loss)
plt.title('Model Loss (Average across folds)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

print('\nFinal Results:')
print(f'Average Validation Accuracy: {np.mean(fold_scores)*100:.2f}%')
print(f'Standard Deviation: {np.std(fold_scores)*100:.2f}%') 