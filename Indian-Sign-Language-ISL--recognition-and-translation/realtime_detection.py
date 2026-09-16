import cv2
import numpy as np
from predict import which
import time
from variables import IMAGE_SIZE

def segment_hand(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)
    
    return mask

def preprocess_roi(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask of the largest contour
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        
        # Apply the mask to the original ROI
        result = cv2.bitwise_and(gray, gray, mask=mask)
        
        return result
    return gray

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Set the width and height of the capture frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Region of Interest (ROI) coordinates - make it square and centered
    roi_size = IMAGE_SIZE * 4  # Make ROI 4 times larger than final size for better detail
    roi_top = (480 - roi_size) // 2  # Center vertically
    roi_bottom = roi_top + roi_size
    roi_left = (640 - roi_size) // 2  # Center horizontally
    roi_right = roi_left + roi_size
    
    prediction_text = ""
    confidence = 0
    last_prediction_time = time.time()
    prediction_interval = 0.5  # Make prediction every 0.5 seconds
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for natural movement
        frame = cv2.flip(frame, 1)
        
        # Extract ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Preprocess ROI for better hand detection
        processed_roi = preprocess_roi(roi)
        
        # Make prediction every 0.5 seconds
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            try:
                confidence, prediction_text = which(roi)  # Use original ROI for prediction
                last_prediction_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        
        # Display prediction and confidence
        cv2.putText(frame, f"Sign: {prediction_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show help text
        cv2.putText(frame, "Place hand in green box", (roi_left, roi_top - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the main frame and processed ROI
        cv2.imshow('Indian Sign Language Detection', frame)
        cv2.imshow('Processed Hand', processed_roi)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 