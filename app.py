import cv2
import mediapipe as mp
import numpy as np
import time
import pymongo
import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()
app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") 
DB_NAME = "FitnessTracker"
COLLECTION_NAME = "SideLegRaises"
MODEL_PATH = 'pose_landmarker_heavy.task'

try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    collection.create_index("user_id", unique = True)
    print("Connected to MongoDB & Index Verified!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

base_options = python.BaseOptions(model_asset_path = MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.VIDEO,
    num_poses = 1,
    min_pose_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_segmentation_masks = False
)
landmarker = vision.PoseLandmarker.create_from_options(options)

class UserSession:
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.is_active = False
        self.start_time = None
        self.feedback = "Press START to begin"
        self.current_angle = 0

sessions = {}

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def process_frame(frame, user_id):
    if user_id not in sessions:
        sessions[user_id] = UserSession()
    session = sessions[user_id]
    
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    
    h, w, _ = frame.shape
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
        
        torso = calculate_angle(landmarks[11], landmarks[23], landmarks[24])
        knee_r = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        session.current_angle = int(knee_r)

        if session.is_active:
            if torso < 20: 
                session.feedback = "Lie straighter!"
            elif knee_r < 100: 
                session.feedback = "Keep leg straight!"
            else:
                session.feedback = "Form Good"
                if knee_r > 165: 
                    session.stage = "down"
                if knee_r < 140 and session.stage == "down":
                    session.stage = "up"
                    session.counter += 1
        else:
            session.feedback = "Paused"

    cv2.rectangle(frame, (0, 0), (w, 80), (245, 117, 16), -1)
    cv2.putText(frame, f'REPS: {session.counter}', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    elapsed = int(time.time() - session.start_time) if session.is_active and session.start_time else 0
    cv2.putText(frame, f'TIME: {elapsed}s', (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.putText(frame, session.feedback, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return frame

def generate_frames(user_id):
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            frame = process_frame(frame, user_id)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/", response_class = HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Fitness Trainer</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; text-align: center; background: #222; color: white; margin: 0; padding: 20px; }
            
            .container { max_width: 1200px; margin: 0 auto; }
            
            .input-group { background: #333; padding: 20px; border-radius: 10px; margin-bottom: 20px; display: inline-block; }
            input { padding: 10px; border-radius: 5px; border: none; }
            
            .video-wrapper { 
                position: relative; 
                margin: 0 auto; 
                background: #000; 
                border-radius: 8px; 
                overflow: hidden; 
                transition: all 0.3s ease;
                border: 2px solid #555;
            }

            #videoStream { width: 100%; height: 100%; object-fit: contain; display: none; }

            .size-small { width: 400px; height: 300px; }
            .size-medium { width: 800px; height: 600px; }
            .size-large { width: 100%; max-width: 1200px; height: auto; }
            
            button { padding: 10px 20px; font-weight: bold; cursor: pointer; border: none; border-radius: 5px; margin: 5px; transition: 0.2s; }
            .btn-blue { background: #007bff; color: white; }
            .btn-green { background: #28a745; color: white; }
            .btn-red { background: #dc3545; color: white; }
            .btn-gray { background: #6c757d; color: white; }
            button:hover { opacity: 0.8; }

            .controls-bar { margin-top: 15px; padding: 10px; background: #333; border-radius: 8px; display: inline-block; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Side Leg Raise Tracker</h1>
            
            <div class="input-group">
                <label>User ID: </label>
                <input type="text" id="userId" placeholder="Enter Name">
                <button class="btn-blue" onclick="loadVideo()">Connect Camera</button>
            </div>

            <div id="videoContainer" class="video-wrapper size-medium">
                <img id="videoStream" src="" alt="Camera Feed">
            </div>

            <div class="controls-bar">
                <button class="btn-gray" onclick="resizeVideo('small')">Small</button>
                <button class="btn-gray" onclick="resizeVideo('medium')">Medium</button>
                <button class="btn-gray" onclick="resizeVideo('large')">Large</button>
                <button class="btn-blue" onclick="toggleFullscreen()">⛶ Fullscreen</button>
                <br><br>
                <button class="btn-green" onclick="controlExercise('start')">▶ START</button>
                <button class="btn-red" onclick="controlExercise('stop')">⏹ STOP & SAVE</button>
            </div>
            
            <p id="statusMsg" style="margin-top: 10px; color: yellow;"></p>
        </div>

        <script>
            let currentUserId = "";

            function loadVideo() {
                currentUserId = document.getElementById("userId").value;
                if(!currentUserId) { alert("Please enter a User ID"); return; }
                
                const img = document.getElementById("videoStream");
                img.src = "/video_feed?user_id=" + currentUserId;
                img.style.display = "block";
                document.getElementById("statusMsg").innerText = "Camera Connected for " + currentUserId;
            }

            function resizeVideo(size) {
                const container = document.getElementById("videoContainer");
                container.className = "video-wrapper"; // Reset classes
                container.classList.add("size-" + size);
            }

            function toggleFullscreen() {
                const elem = document.getElementById("videoStream");
                if (elem.requestFullscreen) {
                    elem.requestFullscreen();
                } else if (elem.webkitRequestFullscreen) { /* Safari */
                    elem.webkitRequestFullscreen();
                } else if (elem.msRequestFullscreen) { /* IE11 */
                    elem.msRequestFullscreen();
                }
            }

            async function controlExercise(action) {
                if(!currentUserId) { alert("Enter User ID first"); return; }
                const response = await fetch(`/${action}?user_id=${currentUserId}`, { method: 'POST' });
                const data = await response.json();
                document.getElementById("statusMsg").innerText = data.message;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed(user_id : str):
    return StreamingResponse(generate_frames(user_id), media_type = "multipart/x-mixed-replace; boundary=frame")

@app.post("/start")
async def start_exercise(user_id : str):
    if user_id not in sessions: sessions[user_id] = UserSession()
    sessions[user_id].is_active = True
    sessions[user_id].start_time = time.time()
    sessions[user_id].counter = 0
    return {"message" : "Started!"}

@app.post("/stop")
async def stop_exercise(user_id : str):
    if user_id not in sessions or not sessions[user_id].is_active:
        return {"message" : "No active session."}
    
    session = sessions[user_id]
    current_duration = round(time.time() - session.start_time, 2)
    current_reps = session.counter
    timestamp = datetime.now()

    collection.update_one(
        {"user_id" : user_id}, 
        {
            "$setOnInsert" : {
                "created_at" : timestamp,
            },
            "$inc" : {
                "total_reps" : current_reps,
                "total_duration" : current_duration
            },
            "$push" : {
                "session_history" : {
                    "date" : timestamp,
                    "reps" : current_reps,
                    "duration" : current_duration
                }
            },
            "$set" : {
                "last_updated": timestamp
            }
        },
        upsert = True 
    )
    
    session.is_active = False
    return {"message" : f"Saved! Added {current_reps} reps to your history."}

@app.post("/progress_report")
async def report(user_id: str):
    record = collection.find_one({"user_id" : user_id}, {"_id" : 0})
    
    if not record:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = f"No record found for user: {user_id}"
        )

    return {
        "user" : user_id,
        "summary" : {
            "total_reps_all_time" : record.get("total_reps", 0),
            "total_duration_seconds" : record.get("total_duration", 0)
        },
        "history" : record.get("session_history", [])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)