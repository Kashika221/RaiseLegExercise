# AI Side Leg Raise Tracker

An AI-powered fitness application that monitors "Side Leg Raise" exercises in real-time. It uses Computer Vision to correct your form, count repetitions, and track session duration, saving all performance data to a MongoDB database.

## Features

* **Real-time Pose Detection:** Uses MediaPipe to track body landmarks with high precision.
* **Form Feedback:** Detects if your torso is too bent or if your leg isn't straight and provides instant text feedback.
* **Automatic Rep Counting:** intelligently counts reps based on leg angle thresholds.
* **Web Interface:** A clean, responsive dashboard built with FastAPI and HTML/CSS.
* **Database Integration:** Automatically logs user sessions (User ID, Reps, Duration, Date) to MongoDB.
* **Session Management:** Supports multiple user IDs and manages states independently.

## Tech Stack

* **Language:** Python 3.8+
* **Web Framework:** FastAPI
* **Computer Vision:** OpenCV, MediaPipe
* **Database:** MongoDB (PyMongo)
* **Frontend:** HTML5, CSS3, JavaScript (Jinja2 templates)

## Project Structure


```

├── main.py                    # Main FastAPI application and logic
├── pose_landmarker_heavy.task # MediaPipe Model file (Required)
├── .env                       # Environment variables (MongoDB URI)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

## Installation

### 1. Prerequisites
* Python installed (version 3.9 or higher recommended).
* MongoDB installed locally or a connection string for MongoDB Atlas.

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/leg-raise-tracker.git](https://github.com/yourusername/leg-raise-tracker.git)
cd leg-raise-tracker

```

### 3. Install Dependencies

Create a virtual environment (optional but recommended) and install the required packages:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install fastapi uvicorn opencv-python mediapipe pymongo python-dotenv jinja2 python-multipart

```

### 4. Download MediaPipe Model

You must download the heavy pose landmarker model to the root directory of the project.

* **Download Link:** [pose_landmarker_heavy.task](https://www.google.com/search?q=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task)
* *Ensure the file is named exactly `pose_landmarker_heavy.task` and is in the same folder as `main.py`.*

### 5. Configure Environment

Create a `.env` file in the root directory:

```bash
MONGO_URI="mongodb://localhost:27017/"

```

*(If using MongoDB Atlas, replace the URI with your connection string).*

## How to Run

1. Start the FastAPI server:
```bash
python main.py

```


*Alternatively, using uvicorn directly:* `uvicorn main:app --reload`
2. Open your browser and navigate to:
```
http://localhost:8000

```



## User Guide

1. **Enter User ID:** Type a unique name or ID (e.g., `user_john`) in the input box.
2. **Connect Camera:** Click the "Connect Camera" button. The video feed should appear.
3. **Setup:** Stand back until your full body (especially legs) is visible.
4. **Start:** Click **"▶ START"**. The app is now tracking.
* *Lie on your side.*
* *Keep your torso straight and raise your top leg.*


5. **Feedback:** Read the on-screen text for corrections (e.g., "Lie straighter", "Keep leg straight").
6. **Stop:** When finished, click **"⏹ STOP & SAVE"**. This stops the timer and saves your stats to MongoDB.

## Database Schema

The data is stored in the `FitnessTracker` database under the `SideLegRaises` collection:

```json
{
  "_id": "65a9f...",
  "user_id": "john_doe",
  "exercise": "Side Leg Raise",
  "reps": 15,
  "duration_seconds": 45.2,
  "timestamp": "2024-01-18T10:30:00.000Z",
  "date_str": "2024-01-18 10:30:00"
}

```

## Troubleshooting

* **Camera not opening?** Ensure no other app (Zoom, Teams) is using the webcam.
* **Video laggy?** Lighting is key for MediaPipe. Ensure the room is well-lit.
* **MongoDB Error?** Check if the MongoDB service is running locally (`mongod`) or check your internet connection if using Atlas.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)