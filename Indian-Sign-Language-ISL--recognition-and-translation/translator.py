import cv2
import numpy as np
import pyttsx3
from predict import *  # Assuming this contains your model and LABELS

class ISLRecognizer:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 105)
        self.engine.setProperty('voice', 1)
        
        # Video capture settings
        self.cap = cv2.VideoCapture(0)
        self.window_name = "ISL Recognition System"
        
        # Frame dimensions
        self.frame_height, self.frame_width = 480, 900
        self.roi_height, self.roi_width = 200, 200
        self.x_start, self.y_start = 100, 100
        
        # Text processing
        self.sentence = ""
        self.prev_label = ""
        self.THRESHOLD = 0.7  # Confidence threshold
        
        # Initialize UI
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        
    def process_hand_image(self, hand_img):
        """Process the hand image for prediction"""
        # Convert color space and apply blur
        img_ycrcb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2YCR_CB)
        blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)
        
        # Skin color detection
        skin_ycrcb_min = np.array((0, 138, 67))
        skin_ycrcb_max = np.array((255, 173, 133))
        mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
        
        # Morphological operations
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Apply mask
        processed_hand = cv2.bitwise_and(hand_img, hand_img, mask=mask)
        return processed_hand, mask
    
    def predict_letter(self, processed_hand):
        """Predict the ISL letter from processed hand image"""
        gray_hand = cv2.cvtColor(processed_hand, cv2.COLOR_BGR2GRAY)
        gray_hand = cv2.resize(gray_hand, (50, 50))
        gray_hand = gray_hand.reshape(50, 50, 1)
        gray_hand = gray_hand / 255.0
        gray_hand = np.expand_dims(gray_hand, axis=0)
        
        preds = model.predict(gray_hand)
        conf = preds.max()
        label = LABELS[int(np.argmax(preds))]
        return label, conf
    
    def handle_key_commands(self, key):
        """Process keyboard commands"""
        # Speak the sentence
        if len(self.sentence) > 0 and key == ord('s'):
            self.engine.say(self.sentence)
            self.engine.runAndWait()
        
        # Clear the sentence
        elif key == ord('c') or key == ord('C'):
            self.sentence = ""
        
        # Delete last character
        elif key == ord('d') or key == ord('D'):
            self.sentence = self.sentence[:-1]
        
        # Add space
        elif key == ord('m') or key == ord('M'):
            self.sentence += " "
        
        # Add predicted letter
        elif key == ord('n') or key == ord('N'):
            if len(self.sentence) == 0 or self.sentence[-1] != self.prev_label:
                self.sentence += self.prev_label
    
    def run(self):
        """Main recognition loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No Frame Captured")
                continue
            
            # Draw ROI rectangle
            cv2.rectangle(frame, 
                         (self.x_start, self.y_start), 
                         (self.x_start + self.roi_width, self.y_start + self.roi_height), 
                         (255, 0, 0), 3)
            
            # Crop hand region
            hand_img = frame[self.y_start:self.y_start + self.roi_height, 
                           self.x_start:self.x_start + self.roi_width]
            
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Process hand image
                processed_hand, mask = self.process_hand_image(hand_img)
                
                # Show processing steps
                cv2.imshow("Hand Mask", mask)
                cv2.imshow("Processed Hand", processed_hand)
                
                # Predict letter
                label, conf = self.predict_letter(processed_hand)
                
                if label != self.prev_label:
                    print(f"Predicted: {label}, Confidence: {conf:.2f}")
                    self.prev_label = label
                
                if conf >= self.THRESHOLD:
                    cv2.putText(frame, label, (50, 50), 
                               cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))
            
            # Display current sentence
            cv2.putText(frame, self.sentence, (50, 70), 
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))
            
            # Show main frame
            cv2.imshow(self.window_name, frame)
            
            # Handle key commands
            key = cv2.waitKey(1) & 0xff
            self.handle_key_commands(key)
            
            # Exit on ESC
            if key == 27:
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = ISLRecognizer()
    recognizer.run()