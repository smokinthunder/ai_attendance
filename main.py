import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pickle

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_file = "attendance.csv"
        self.encodings_file = "face_encodings.pkl"
        self.today_attendance = set()
        
        # Initialize face recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_id_map = {}  # Map face IDs to names
        
        # Load existing face data
        self.load_face_data()
        
        # Initialize CSV file
        self.init_csv()
    
    def load_face_data(self):
        """Load face data from files"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_id_map = data.get('face_id_map', {})
                print(f"Loaded {len(self.face_id_map)} known faces")
                
                # Load trained recognizer
                recognizer_file = "face_recognizer.yml"
                if os.path.exists(recognizer_file):
                    self.recognizer.read(recognizer_file)
                    print("Face recognizer model loaded")
            except Exception as e:
                print(f"Error loading face data: {e}")
        else:
            print("No existing face data found. Please add faces first.")
    
    def save_face_data(self):
        """Save face data to files"""
        try:
            data = {'face_id_map': self.face_id_map}
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save trained recognizer
            recognizer_file = "face_recognizer.yml"
            self.recognizer.save(recognizer_file)
            print("Face data saved!")
        except Exception as e:
            print(f"Error saving face data: {e}")
    
    def add_face_from_camera(self, name):
        """Add a new face by capturing from camera"""
        print(f"Adding face for {name}")
        print("Look at the camera and press SPACE to capture, ESC to cancel")
        
        cap = cv2.VideoCapture(0)
        face_id = len(self.face_id_map) + 1
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Press SPACE to capture for {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Add Face - Press SPACE to capture, ESC to cancel', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key
                for (x, y, w, h) in faces:
                    count += 1
                    # Save face sample
                    face_dir = "face_samples"
                    if not os.path.exists(face_dir):
                        os.makedirs(face_dir)
                    
                    cv2.imwrite(f"{face_dir}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
                    print(f"Captured sample {count}")
                    
                    if count >= 30:  # Capture 30 samples
                        break
                
                if count >= 30:
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            self.face_id_map[face_id] = name
            self.train_recognizer()
            print(f"Successfully added {name} with {count} samples")
            return True
        else:
            print("No face samples captured")
            return False
    
    def add_face_from_image(self, image_path, name):
        """Add a face from an image file"""
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found!")
            return False
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            return False
        
        face_id = len(self.face_id_map) + 1
        self.face_id_map[face_id] = name
        
        # Create face samples directory
        face_dir = "face_samples"
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        
        # Save multiple samples from the image
        for i, (x, y, w, h) in enumerate(faces[:1]):  # Use first face only
            face_img = gray[y:y+h, x:x+w]
            
            # Create variations by resizing and adjusting
            for j in range(10):
                # Add slight variations
                variation = cv2.resize(face_img, (100, 100))
                if j % 2 == 0:
                    variation = cv2.flip(variation, 1)  # Horizontal flip
                
                cv2.imwrite(f"{face_dir}/User.{face_id}.{j+1}.jpg", variation)
        
        self.train_recognizer()
        print(f"Successfully added {name}")
        return True
    
    def train_recognizer(self):
        """Train the face recognizer with collected samples"""
        face_dir = "face_samples"
        if not os.path.exists(face_dir):
            print("No face samples found!")
            return
        
        faces = []
        ids = []
        
        for filename in os.listdir(face_dir):
            if filename.endswith('.jpg'):
                try:
                    # Extract face ID from filename
                    face_id = int(filename.split('.')[1])
                    img_path = os.path.join(face_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        faces.append(img)
                        ids.append(face_id)
                except (ValueError, IndexError):
                    continue
        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.save_face_data()
            print(f"Training completed with {len(faces)} samples")
        else:
            print("No valid face samples found for training")
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Status'])
    
    def mark_attendance(self, name):
        """Mark attendance for a person"""
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        # Check if already marked today
        if name in self.today_attendance:
            return False
        
        # Add to today's attendance
        self.today_attendance.add(name)
        
        # Write to CSV
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_string, time_string, 'Present'])
        
        print(f"✓ Attendance marked for {name} at {time_string}")
        return True
    
    def run_attendance(self):
        """Main function to run attendance system"""
        if len(self.face_id_map) == 0:
            print("No known faces found! Please add faces first.")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting attendance system. Press 'q' to quit.")
        print(f"Known faces: {', '.join(self.face_id_map.values())}")
        
        # Reset today's attendance
        self.today_attendance.clear()
        
        # Recognition parameters
        confidence_threshold = 100  # Lower values mean more confident recognition
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                
                # Recognize face
                face_id, confidence = self.recognizer.predict(face_gray)
                
                # Determine name based on confidence
                if confidence < confidence_threshold and face_id in self.face_id_map:
                    name = self.face_id_map[face_id]
                    color = (0, 255, 0)  # Green for recognized
                    
                    # Mark attendance
                    if self.mark_attendance(name):
                        pass  # Attendance marked message already printed
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and name
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
                cv2.putText(frame, f"{name}", (x+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show confidence for debugging
                if name != "Unknown":
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x+5, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display attendance count
            cv2.putText(frame, f"Present Today: {len(self.today_attendance)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show present names
            y_offset = 60
            for name in self.today_attendance:
                cv2.putText(frame, f"✓ {name}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Show frame
            cv2.imshow('Attendance System - Press Q to quit', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nAttendance session completed!")
        print(f"Total present: {len(self.today_attendance)}")
        if self.today_attendance:
            print(f"Present: {', '.join(self.today_attendance)}")

# Utility functions
def view_attendance():
    """View today's attendance"""
    if not os.path.exists("attendance.csv"):
        print("No attendance file found!")
        return
    
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nAttendance for {today}:")
    print("-" * 40)
    
    with open("attendance.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        found_today = False
        for row in reader:
            if len(row) >= 4 and row[1] == today:
                print(f"{row[0]} - {row[2]} ({row[3]})")
                found_today = True
        
        if not found_today:
            print("No attendance records found for today.")

def setup_faces():
    """Setup function to add faces easily"""
    system = AttendanceSystem()
    
    while True:
        print("\n=== Face Setup ===")
        print("1. Add face from camera")
        print("2. Add face from image file")
        print("3. List known faces")
        print("4. Exit setup")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter person's name: ").strip()
            if name:
                system.add_face_from_camera(name)
        
        elif choice == '2':
            name = input("Enter person's name: ").strip()
            image_path = input("Enter image path: ").strip()
            if name and image_path:
                system.add_face_from_image(image_path, name)
        
        elif choice == '3':
            if system.face_id_map:
                print("Known faces:")
                for face_id, name in system.face_id_map.items():
                    print(f"  {face_id}: {name}")
            else:
                print("No faces registered yet.")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice!")

# Example usage
if __name__ == "__main__":
    print("=== Attendance System ===")
    print("1. Setup faces (first time)")
    print("2. Run attendance")
    print("3. View today's attendance")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == '1':
        setup_faces()
    
    elif choice == '2':
        system = AttendanceSystem()
        system.run_attendance()
    
    elif choice == '3':
        view_attendance()
    
    else:
        print("Invalid choice!")

