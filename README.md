
# 🎓 Face Recognition Attendance System

This project implements a real-time **Face Recognition Attendance System** using Python. Leveraging **OpenCV**, **FaceNet**, and **YuNet**, it captures facial data from a webcam, recognizes individuals based on trained data, marks attendance in a CSV/Excel file, and can even email the attendance report.

---

## 👩‍💻 Developed By

- Neha Golani 

---

## 🚀 Features

- 💻 Real-time face detection and recognition via webcam
- 🧠 Embedding extraction using `keras-facenet`
- 🎯 Classification using Support Vector Machine (SVM)
- 📥 Attendance logging with name, time, date, and confidence
- 📊 Export attendance logs to Excel format with summary sheets
- 📧 Send attendance reports directly via email

---

## 🛠 Technologies Used

- Python
- OpenCV
- keras-facenet
- scikit-learn
- NumPy & Pandas
- YuNet (ONNX model for face detection)
- joblib (model persistence)
- smtplib (email)

---

## 🗂️ Directory Structure

```
├── face_detection_yunet_2023mar.onnx        # YuNet ONNX model
├── Dataset/
│   └── Faces/
│       ├── Alice/
│       │   ├── Alice_0.jpg
│       └── Bob/
│           ├── Bob_0.jpg
├── face_recognition_model.pkl               # Trained SVM model
├── label_encoder.pkl                        # Label encoder
├── attendance.csv                           # Attendance log
├── attendance.xlsx                          # Exported report
└── FaceRecognitionAttendance.py             # Main Python script
```

---

## 📸 How It Works

1. **Collect Face Data**  
   Capture 20+ face images for each person using your webcam.

2. **Train Model**  
   FaceNet generates 512-dimensional embeddings for each face, then SVM is trained on this data.

3. **Start Attendance**  
   The system detects and recognizes faces in real-time, then logs entries in `attendance.csv`.

4. **Export to Excel**  
   The log can be converted into an Excel report with a daily summary.

5. **Email Attendance**  
   Automatically email the report using Gmail SMTP.

---

## 🧪 How to Use

1. **Install Dependencies**

```bash
pip install opencv-python keras-facenet scikit-learn numpy pandas matplotlib joblib xlsxwriter
```

2. **Place the YuNet ONNX model**

Download `face_detection_yunet_2023mar.onnx` and place it in the project root.

3. **Run the Script**

```bash
python FaceRecognitionAttendance.py
```

4. **Follow Menu Options**

```
1. Collect Data for New Person
2. Train Model
3. Start Attendance System
4. Export Attendance to Excel
5. Send Mail into Your Account
6. Exit
```

---

## 🔐 Notes

- You must enable **2-Step Verification** and generate an **App Password** to send emails from your Gmail account.
- Adjust the face recognition confidence threshold in the code if needed (`threshold = 0.7`).

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📬 Contact

For questions or collaboration:

- Neha Golani: 22bec042@nirmauni.ac.in  
---
