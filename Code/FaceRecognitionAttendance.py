import os
import cv2
import smtplib
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from email.message import EmailMessage
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt

# Install the keras-facenet package to avoid the EOFError
# pip install keras-facenet mtcnn opencv-python

# Import FaceNet from keras-facenet package
from keras_facenet import FaceNet

# Path to YuNet model
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"  # path to the YuNet model

# Initialize FaceNet model using keras-facenet (this avoids the EOFError)
embedder = FaceNet()

# Load the YuNet face detector
face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320))

def detect_face(img):
    """Detect faces using YuNet and return the face ROI."""   #region in interest specific portion of the image 
    h, w = img.shape[:2]                       # take size if image 
    face_detector.setInputSize((w, h))         #set input size 
    _, faces = face_detector.detect(img)  # Correct unpacking of the detect method
    
    if faces is not None and len(faces) > 0:      # ensure facelist is nor empty
        # YuNet returns [x, y, w, h, ...] for each face
        x, y, w, h = faces[0][:4].astype(int)         
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        w = min(w, img.shape[1] - x)         #range vadhare na javu joiye so that use this min 
        h = min(h, img.shape[0] - y)
        return img[y:y+h, x:x+w]                
    return None      #if no face detected

def get_embedding(face_img):
    """Get embedding from a face image using FaceNet."""
    # Preprocess the image to 160x160 as expected by FaceNet
    face_img = cv2.resize(face_img, (160, 160))
    # Use keras-facenet to get the embeddings
    face_array = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_array)[0]  # Get the 512-dimensional embedding

def create_embeddings(dataset_path):
    """Process all images in the dataset and create embeddings."""
    embeddings = []     #numerical value of faces
    labels = []        #store the name of people 
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path): 
        print(f"Dataset path does not exist: {dataset_path}")
        return np.array([]), []
        
    for person in os.listdir(dataset_path):        #return list of all entries
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
            
        print(f"Processing images for: {person}")
        
        # Process each image for this person
        image_count = 0
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            
            # Skip non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:       # if fine type png , jpg, jpeg na hoy to 
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                # Detect face using YuNet
                face = detect_face(img)
                if face is not None:
                    # Get embedding
                    emb = get_embedding(face)
                    embeddings.append(emb)
                    labels.append(person)
                    image_count += 1
                else:
                    print(f"No face detected in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        print(f"Processed {image_count} images for {person}")
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    return embeddings, labels

def train_classifier(embeddings, labels):
    """Train an SVM classifier on face embeddings."""     #docstring for better documentation purpose 
    if len(embeddings) == 0 or len(labels) == 0:     # jyare record na hoy tyare empty list 
        print("No data to train classifier!")
        return None, None
        
    # Encode the string labels to integers
    le = LabelEncoder()    # it is a sklearn lib ( string ----------> numerical ) bcz sometime model can not take string value like svm...
    labels = le.fit_transform(labels)     # transform ghani bdhi image mate useful che 
    
    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)    #for classification and true is for confidence level
    clf.fit(embeddings, labels)    # it is used to set the parameter so that we can get proper classification 
    
    print(f"Classifier trained with {len(np.unique(labels))} classes")
    return clf, le

def save_models(clf, le, classifier_path, label_encoder_path):
    """Save the trained classifier and label encoder."""
    if clf is not None and le is not None:
        joblib.dump(clf, classifier_path)       # save the clf and le at specific path joblib s a python function 
        joblib.dump(le, label_encoder_path)
        print(f"Models saved to {classifier_path} and {label_encoder_path}")
    else:
        print("Cannot save models - they are None")

def load_models(classifier_path, label_encoder_path):
    """Load the trained classifier and label encoder."""
    if os.path.exists(classifier_path) and os.path.exists(label_encoder_path):
        clf = joblib.load(classifier_path)     # actually load into a disk
        le = joblib.load(label_encoder_path)
        print("Models loaded successfully")
        return clf, le
    else:
        print("Model files not found!")
        return None, None

def ensure_file_exists(file_path, headers):
    """Create file with headers if it doesn't exist."""
    if not os.path.exists(file_path):         #create file is not exist 
        with open(file_path, 'w') as f:
            f.write(headers + '\n')            # create file in write mode 

def log_attendance(name, confidence, attendance_file="attendance.csv"):
    """Log attendance with timestamp to CSV file."""
    # Ensure the file exists with headers
    ensure_file_exists(attendance_file, "Name,Date,Time,Confidence")
    
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")   #current date 
    time_str = now.strftime("%H:%M:%S")   #current time
    
    # Check if this person has already been marked today
    df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame()
    
    # If this person is already marked today, don't mark again
    if not df.empty and len(df[(df['Name'] == name) & (df['Date'] == date_str)]) > 0:
        print(f"{name} already marked for today!")
        return False
    
    # Append the attendance record
    with open(attendance_file, 'a') as f:   # open in append mode 
        f.write(f"{name},{date_str},{time_str},{confidence:.2f}\n")
    
    print(f"Attendance marked for {name}")
    return True

def recognize_face(img, clf, le):
    """Recognize a face in the image using the trained classifier."""
    if clf is None or le is None:
        return "Models not loaded", 0
        
    # Detect face using YuNet
    face = detect_face(img)
    if face is None:
        return "No face detected", 0
    
    # Get embedding
    emb = get_embedding(face)
    
    # Predict with probability
    proba = clf.predict_proba([emb])[0]
    max_proba = max(proba)
    predicted_idx = np.argmax(proba)
    
    # Get the name from the label encoder    numerical value --------------> original name 
    predicted_name = le.inverse_transform([predicted_idx])[0]
    
    # Only return the name if the confidence is above threshold
    threshold = 0.7  # You can adjust this threshold  if below 70 predict ad unknown 
    if max_proba >= threshold:
        return predicted_name, max_proba
    else:
        return "Unknown", max_proba

def collect_face_data(output_dir, person_name, num_images=20):
    """Collect face images from webcam for training."""
    # Create directory if it doesn't exist
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened!")
        return
    
    count = 0
    while count < num_images:    #start counter to collect images 
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image!")
            break
        
        # Show frame with info
        info_frame = frame.copy()   # create a copy of the frame 
        cv2.putText(     # draw text on the image 
            info_frame, 
            f"Capturing: {count}/{num_images} for {person_name}",    # text to on display
            (10, 30),                                                #x,y position on image 
            cv2.FONT_HERSHEY_SIMPLEX,                                #font type   like FONT_HERSHEY_COMPLEX....
            0.7,                                                     #font size 
            (0, 255, 0),                                             # colour (it is green)  in bgr ( blue, green , red) ex for purple (255,0,255)
            2                                                        #thickness of words
        )
        cv2.imshow("Capturing", info_frame)
        
        key = cv2.waitKey(1)
        
        # Press 'c' to capture
        if key == ord('c'):     # ord give the ascii value of the c key 
            face = detect_face(frame)
            if face is not None:
                # Save the face image
                img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                cv2.imwrite(img_path, face)
                print(f"Saved {img_path}")
                count += 1
            else:
                print("No face detected in frame")
        
        # Press 'q' to quit
        elif key == ord('q'):
            break
    
    cap.release()          # stop the camera 
    cv2.destroyAllWindows()     # close opnecv window 
    print(f"Collected {count} images for {person_name}")

def test_with_camera(clf, le, attendance_file="attendance.csv"):
    """Run face recognition on webcam feed and mark attendance."""
    if clf is None or le is None:
        print("Models not loaded! Cannot continue.")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened!")
        return
    
    print("Starting camera feed. Press 'q' to quit.")
    
    # For displaying recognition status
    recognition_status = None
    status_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image!")
            break
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect and recognize face
        face = detect_face(frame)
        if face is not None:
            # Draw rectangle around face
            h, w = frame.shape[:2]     #retrive height and width 
            face_detector.setInputSize((w, h))  
            _, faces = face_detector.detect(frame)        #give number of detected face 
            
            if faces is not None and len(faces) > 0:
                # Get coordinates for drawing
                x, y, w, h = faces[0][:4].astype(int)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   # draw green rectaguler around detected face with thickness 2
                
                # Recognize face
                name, confidence = recognize_face(frame, clf, le) 
                
                # Display name and confidence
                text = f"{name} ({confidence:.2f})"
                cv2.putText(
                    display_frame, 
                    text, 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Press 'm' to mark attendance
                if cv2.waitKey(1) & 0xFF == ord('m'):    # wait key is pressed or not 
                    if name != "Unknown":
                        marked = log_attendance(name, confidence, attendance_file)    #store attendance into a csv file 
                        if marked:
                            recognition_status = f"Attendance marked for {name}"
                        else:
                            recognition_status = f"{name} already marked today"
                        status_time = datetime.now()
        
        # Display status message if it exists
        if recognition_status and (datetime.now() - status_time).total_seconds() < 3:  #if time is less than 3s give the msg 
            cv2.putText(
                display_frame, 
                recognition_status, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        
        # Instructions
        cv2.putText(
            display_frame,
            "Press 'm' to mark attendance, 'q' to quit",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.imshow("Face Recognition Attendance", display_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def export_attendance_to_excel(csv_file="attendance.csv", excel_file="attendance.xlsx"):
    """Export the attendance CSV to Excel format with summary."""
    if not os.path.exists(csv_file):
        print(f"Attendance file {csv_file} does not exist!")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Create Excel writer
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')  #tells pandas to use excelwriter engine to write the excel fine 
    
    # Write the detailed data
    df.to_excel(writer, sheet_name='Detailed', index=False)
    
    # Create a summary by date
    summary = df.groupby(['Date', 'Name']).size().unstack(fill_value=0)
    summary.to_excel(writer, sheet_name='Summary')
    
    # Save the Excel file
    writer._save()
    print(f"Attendance exported to {excel_file}")


def delete_existing_data():
    attendance_csv = "attendance.csv"
    attendance_xlsx = "attendance.xlsx"
    
    # Delete model and label encoder
    for file_path in [attendance_csv, attendance_xlsx]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

    print("All selected data deleted.")



def send_attendance_email(sender_email, app_password, receiver_email, file_path):
    """Send the attendance file via email."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    msg = EmailMessage()
    msg['Subject'] = "Attendance Report"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content("Attached is the attendance report.")

    # Read file and attach
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    
    try:
        # Connect to Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
            print(f"Attendance sent to {receiver_email} successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


def main():
    # Define path 
    dataset_path = "Dataset/Faces"  # Directory containing subdirectories with person's images
    clf_path = "face_recognition_model.pkl"  # Path to save/load classifier
    le_path = "label_encoder.pkl"  # Path to save/load label encoder
    attendance_file = "attendance.csv"  # Path for attendance log
    
    # Create necessary directories
    os.makedirs(dataset_path, exist_ok=True)
    
    # Menu system
    while True:
        print("\n===== Face Recognition Attendance System =====")
        print("1. Collect Data for New Person")
        print("2. Train Model")
        print("3. Start Attendance System")
        print("4. Export Attendance to Excel")
        print("5. send mail into your account")
        print("6. Exit")

        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            name = input("Enter person's name: ")
            num_images = int(input("How many images to capture (default 20): ") or "20")
            collect_face_data(dataset_path, name, num_images)
            
        elif choice == '2':
            print("Training model from dataset...")
            embeddings, labels = create_embeddings(dataset_path)
            if len(embeddings) > 0:
                clf, le = train_classifier(embeddings, labels)
                save_models(clf, le, clf_path, le_path)
                print("Model training complete!")
            else:
                print("No face data found in the dataset!")
            
        elif choice == '3':
            print("Starting attendance system...")
            clf, le = load_models(clf_path, le_path)
            test_with_camera(clf, le, attendance_file)
            
        elif choice == '4':
            print("Exporting attendance to Excel...")
            export_attendance_to_excel(attendance_file)


        elif choice == '5':
          sender = input("Enter your email address: ")
          app_password = input("Enter your app password: ")
          receiver = input("Enter receiver's email address: ")
          file_path = input("Enter file to send (default: attendance.xlsx): ") or "attendance.xlsx"
          send_attendance_email(sender, app_password, receiver, file_path)
       
        elif choice == '6':
            print("Exiting program.")
            break

        else:
            print("Invalid choice! Please try again.")

if _name_ == "_main_":
    main()
