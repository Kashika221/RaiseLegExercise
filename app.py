import cv2
import mediapipe as mp
import numpy as np
import time
import pymongo
import os
import base64
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

templates = Jinja2Templates(directory = "templates")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") 
DB_NAME = "FitnessTracker"
COLLECTION_NAME = "SideLegRaises"
MODEL_PATH = 'pose_landmarker_heavy.task'

try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index("user_id", unique = True)
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

base_options = python.BaseOptions(model_asset_path = MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.IMAGE,
    num_poses = 1,
    min_pose_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_segmentation_masks = False
)
landmarker = vision.PoseLandmarker.create_from_options(options)

LEG_RAISE_UP_THRESHOLD = 140      # Leg is raised (angle < this)
LEG_RAISE_DOWN_THRESHOLD = 165    # Leg is lowered (angle > this)
TORSO_ANGLE_MIN = 50             # Ensure user is lying relatively straight
LEFT_LEG_FOLD_THRESHOLD = 180     # Left leg should be folded (angle < this)
RIGHT_LEG_STRAIGHT_THRESHOLD = 50 # Keep right leg straight (angle > this)

class UserSession:
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.is_active = False
        self.start_time = None
        self.feedback = "Press START"
        self.current_right_knee_angle = 0

sessions = {}

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle  
    return angle

def draw_all_landmarks(image, landmarks):
    h, w, _ = image.shape
    for idx, lm in enumerate(landmarks):
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
    connections = [
        (11, 12), (23, 24),  
        (23, 25), (25, 27),  
        (24, 26), (26, 28),  
        (11, 23), (12, 24)   
    ]
    
    for start, end in connections:
        if start < len(landmarks) and end < len(landmarks):
            pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
            pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)

def check_form(landmarks):
    torso_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[24])
    left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    
    feedback_msgs = []
    form_correct = True

    if torso_angle < TORSO_ANGLE_MIN:
        feedback_msgs.append("Lie Straight!")
        form_correct = False
        
    if left_knee_angle > LEFT_LEG_FOLD_THRESHOLD:
        feedback_msgs.append("Fold Left Leg Back")
        form_correct = False

    if right_knee_angle < RIGHT_LEG_STRAIGHT_THRESHOLD:
        feedback_msgs.append("Straighten Right Leg")
        form_correct = False
        
    return form_correct, feedback_msgs, right_knee_angle

def process_image(frame, user_id):
    if user_id not in sessions:
        sessions[user_id] = UserSession()
    session = sessions[user_id]
    
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = landmarker.detect(mp_image)
    h, w, _ = frame.shape
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        draw_all_landmarks(frame, landmarks)
        
        if session.is_active:
            form_correct, feedback_list, right_knee_angle = check_form(landmarks)
            session.current_right_knee_angle = int(right_knee_angle)

            if not form_correct:
                session.feedback = feedback_list[0] if feedback_list else "Adjust Form"
            else:
                if right_knee_angle > LEG_RAISE_DOWN_THRESHOLD:
                    session.stage = "down"
                    session.feedback = "GO UP"
                
                if right_knee_angle < LEG_RAISE_UP_THRESHOLD and session.stage == "down":
                    session.stage = "up"
                    session.counter += 1
                    session.feedback = "GOOD REP!"

                if right_knee_angle < LEG_RAISE_UP_THRESHOLD and session.stage == "up":
                    session.feedback = "Lower Slowly"
                    
        else:
            session.feedback = "Paused - Press START"
            
    cv2.rectangle(frame, (0, 0), (w, 100), (245, 117, 16), -1)
    cv2.putText(frame, f'REPS: {session.counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    elapsed = int(time.time() - session.start_time) if session.is_active and session.start_time else 0
    cv2.putText(frame, f'TIME: {elapsed}s', (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    msg_color = (255, 255, 255)
    if "Straight" in session.feedback or "Fold" in session.feedback:
        msg_color = (0, 0, 255) 
    elif "GOOD" in session.feedback:
        msg_color = (0, 255, 0) 
        
    cv2.putText(frame, session.feedback, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)
    cv2.putText(frame, f"Angle: {session.current_right_knee_angle}", (w - 180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return frame

@app.get("/", response_class = HTMLResponse)
async def index(request : Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket : WebSocket, user_id : str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if "," in data:
                header, encoded = data.split(",", 1)
            else:
                encoded = data
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue
            processed_frame = process_image(frame, user_id)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")
            
    except WebSocketDisconnect:
        print(f"Client #{user_id} disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try : await websocket.close()
        except : pass

@app.post("/start")
async def start_exercise(user_id : str):
    if user_id not in sessions: sessions[user_id] = UserSession()
    sessions[user_id].is_active = True
    sessions[user_id].start_time = time.time()
    sessions[user_id].counter = 0
    sessions[user_id].stage = "down" 
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
            "$setOnInsert" : { "created_at" : timestamp },
            "$inc" : { "total_reps" : current_reps, "total_duration" : current_duration },
            "$push" : { "session_history" : { "date" : timestamp, "reps" : current_reps, "duration" : current_duration } },
            "$set" : { "last_updated" : timestamp }
        },
        upsert = True 
    )
    
    session.is_active = False
    session.feedback = "Session Saved."
    return {"message" : f"Saved! Added {current_reps} reps."}

@app.post("/progress_report")
async def report(user_id : str):
    record = collection.find_one({"user_id" : user_id}, {"_id" : 0})
    if not record:
        raise HTTPException(status_code=404, detail = "No record found")
    return record

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)