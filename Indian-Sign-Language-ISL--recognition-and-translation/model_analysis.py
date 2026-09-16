import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from variables import LABELS

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS,
                yticklabels=LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load the model
    print("Loading model...")
    model = load_model('model.h5')
    
    # Load training data
    print("Loading training data...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    
    # Split data into train and validation sets
    print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Evaluate model on validation data
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_val_classes, y_pred_classes)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val_classes, y_pred_classes, target_names=LABELS))
    
    # If history.npy exists, plot training history
    try:
        print("\nTrying to plot training history...")
        history = np.load('history.npy', allow_pickle=True).item()
        plot_training_history(history)
        print("Training history plots saved as 'training_history.png'")
    except:
        print("\nNo training history found. Could not plot training curves.")
    
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Additional analysis for overfitting/underfitting
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print("\nOverfitting/Underfitting Analysis:")
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    
    if train_accuracy > val_accuracy + 0.1:
        print("\nPossible OVERFITTING detected:")
        print("- Training accuracy is significantly higher than validation accuracy")
        print("- Model might be memorizing training data instead of learning general patterns")
        print("\nSuggestions:")
        print("1. Add dropout layers")
        print("2. Use data augmentation")
        print("3. Reduce model complexity")
        print("4. Add regularization")
    elif train_accuracy < 0.7 and val_accuracy < 0.7:
        print("\nPossible UNDERFITTING detected:")
        print("- Both training and validation accuracy are low")
        print("- Model might be too simple to capture the patterns")
        print("\nSuggestions:")
        print("1. Increase model complexity")
        print("2. Add more training data")
        print("3. Train for more epochs")
        print("4. Reduce regularization if present")
    else:
        print("\nModel seems to be well balanced!")
        print("- Training and validation accuracies are close")
        print("- No clear signs of overfitting or underfitting")

if __name__ == "__main__":
    main() 