# Face Recognition Attendance System

A Python-based attendance system that uses computer vision to automatically mark attendance by recognizing faces through a webcam. The system captures faces, trains a recognition model, and logs attendance to a CSV file.

## Features

- **Real-time Face Recognition**: Automatically detects and recognizes faces using webcam
- **Attendance Logging**: Records attendance with name, date, and time in CSV format
- **Duplicate Prevention**: Prevents marking attendance multiple times for the same person on the same day
- **Interactive Setup**: Easy-to-use interface for adding new faces to the system
- **Cross-platform**: Works on both Windows and Linux
- **Persistent Storage**: Saves trained face models for future use
- **Visual Feedback**: Real-time display with confidence levels and attendance status

## Requirements

### System Requirements
- Python 3.7 - 3.11 (recommended: Python 3.9)
- Webcam/Camera
- Minimum 4GB RAM
- Windows 10+ or Linux (Ubuntu 18.04+)

### Python Package Requirements
- OpenCV 4.5+
- NumPy 1.21+
- Other dependencies listed in `requirements.txt`

## Installation

### Windows Installation

1. **Install Python**:
   - Download Python 3.9 from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Clone or Download the Project**:
   ```cmd
   # If using git
   git clone https://github.com/smokinthunder/ai_attendance.git
   cd ai_attendance
   
   # Or download and extract the ZIP file
   ```

3. **Create Virtual Environment** (Recommended):
   ```cmd
   python -m venv attendance_env
   attendance_env\Scripts\activate
   ```

4. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

5. **Verify Installation**:
   ```cmd
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

### Linux Installation

1. **Install Python and Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   sudo apt install python3-opencv
   
   # For camera access permissions
   sudo usermod -a -G video $USER
   # Log out and log back in after this command
   
   # CentOS/RHEL/Fedora
   sudo dnf install python3 python3-pip python3-opencv
   # or
   sudo yum install python3 python3-pip
   ```

2. **Clone or Download the Project**:
   ```bash
   git clone https://github.com/smokinthunder/ai_attendance.git
   cd ai_attendance
   ```

3. **Create Virtual Environment** (Recommended):
   ```bash
   python3 -m venv attendance_env
   source attendance_env/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**:
   ```bash
   python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

## Usage

### Quick Start

1. **Activate Virtual Environment** (if created):
   ```bash
   # Windows
   attendance_env\Scripts\activate
   
   # Linux
   source attendance_env/bin/activate
   ```

2. **Run the Application**:
   ```bash
   python attendance_system.py
   ```

3. **Follow the Menu**:
   - Choose option 1 for first-time setup (add faces)
   - Choose option 2 to run attendance system
   - Choose option 3 to view today's attendance

### Detailed Usage Instructions

#### Step 1: Initial Setup (Add Faces)

Run the script and select option 1:
```bash
python attendance_system.py
```

**Adding Face from Camera** (Recommended):
1. Select option 1 in the setup menu
2. Enter the person's name
3. Look at the camera when the window opens
4. Press SPACE when ready to capture
5. System will automatically capture 30 samples
6. Press ESC to cancel if needed

**Adding Face from Image File**:
1. Select option 2 in the setup menu
2. Enter the person's name
3. Provide the path to a clear image file (JPG/PNG)
4. Image should contain only one clear face

#### Step 2: Run Attendance System

After adding faces, select option 2 from the main menu:
1. The camera window will open
2. Stand in front of the camera
3. Green rectangle = recognized face (attendance marked)
4. Red rectangle = unknown face
5. Press 'Q' to quit the system

#### Step 3: View Attendance Records

Select option 3 to view today's attendance or use utility functions:
```python
# View today's attendance
view_attendance()

# Export specific date attendance
export_attendance("2024-01-15")
```

## File Structure

```
ai_attendance/
├── main.py      # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── attendance.csv           # Generated attendance records
├── face_encodings.pkl       # Saved face data
├── face_recognizer.yml      # Trained recognition model
└── face_samples/            # Directory for face training samples
    ├── User.1.1.jpg
    ├── User.1.2.jpg
    └── ...
```

## Configuration

### Camera Settings
If the default camera doesn't work, modify the camera index:
```python
# In the code, change:
cap = cv2.VideoCapture(0)  # Try 1, 2, etc. for other cameras
```

### Recognition Sensitivity
Adjust recognition confidence in the `run_attendance()` function:
```python
confidence_threshold = 100  # Lower = more strict, Higher = more lenient
```

### CSV Output Format
The attendance file contains:
- Name: Person's name
- Date: YYYY-MM-DD format
- Time: HH:MM:SS format
- Status: Always "Present"

## Troubleshooting

### Common Issues

**Camera Not Working**:
- Try different camera indices (0, 1, 2)
- Check camera permissions
- Ensure no other application is using the camera
- On Linux, verify user is in the 'video' group

**Poor Recognition Accuracy**:
- Ensure good lighting when adding faces
- Add faces from multiple angles
- Capture faces with different expressions
- Lower the confidence threshold for stricter recognition

**Import Errors**:
```bash
# Reinstall OpenCV packages
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

**Permission Errors on Linux**:
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Then log out and log back in
```

### Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | Install opencv-python: `pip install opencv-python` |
| `Camera not found` | Try different camera index or check camera connection |
| `No faces detected` | Ensure good lighting and face is clearly visible |
| `Permission denied` | Run with appropriate permissions or check camera access |

## Advanced Usage

### Adding Multiple Faces at Once
```python
# Use the setup_faces() function
setup_faces()
```

### Batch Processing Images
```python
system = AttendanceSystem()
faces_dir = "path/to/faces/"
for filename in os.listdir(faces_dir):
    if filename.endswith(('.jpg', '.png')):
        name = filename.split('.')[0]  # Use filename as name
        system.add_face_from_image(os.path.join(faces_dir, filename), name)
```

### Custom Attendance Reports
```python
# Generate custom reports
import pandas as pd
df = pd.read_csv('attendance.csv')
daily_report = df.groupby('Date').count()
print(daily_report)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all requirements are installed
3. Ensure camera is working with other applications
4. Check Python and package versions

## Version History

- **v1.0**: Initial release with basic face recognition
- **v1.1**: Added interactive setup and improved error handling
- **v1.2**: Enhanced recognition accuracy and cross-platform support

---

**Note**: This system is designed for educational and small-scale attendance tracking. For large-scale deployments, consider additional security and privacy measures.