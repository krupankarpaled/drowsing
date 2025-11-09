import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize pygame for alert sound
pygame.mixer.init()

try:
    pygame.mixer.music.load("beep.mp3")
    def play_beep():
        pygame.mixer.music.play()
except:
    import sounddevice as sd
    def play_beep():
        freq = 1000  # 1000 Hz
        duration = 0.3
        t = np.linspace(0, duration, int(44100 * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
        sd.play(tone, 44100)
        sd.wait()

# Streamlit config
st.set_page_config(page_title="Drowsiness Detection Dashboard", layout="wide")

# Custom dark UI
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        [data-testid="stSidebar"] { background-color: #1a1c23; }
        h1, h2, h3 { color: #58a6ff; }
        .stAlert { border-radius: 12px; font-size: 18px; }
        .metric-container {
            background-color: #161a25;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Detection Settings")
EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
EYE_CLOSED_SECONDS = st.sidebar.slider("Closed-Eye Duration (s)", 1.0, 4.0, 2.0, 0.1)
st.sidebar.markdown("---")
st.sidebar.info("👁️ Lower EAR = stricter detection.\nIncrease duration for more tolerance.")

# Header
st.markdown("<h1 style='text-align:center;'>😴 Drowsiness Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Real-time drowsiness detection using OpenCV, MediaPipe & Streamlit.</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR function (use float coordinates)
def eye_aspect_ratio(landmarks, eye_indices):
    p = np.array([[landmarks[i][0], landmarks[i][1]] for i in eye_indices])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Layout
col1, col2 = st.columns([2, 1])
FRAME_WINDOW = col1.image([], channels="RGB", use_container_width=True)
run = col1.toggle("▶️ Start Drowsiness Detection")

with col2:
    st.markdown("<div class='metric-container'><h3>Eye Aspect Ratio (EAR)</h3>", unsafe_allow_html=True)
    ear_display = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='metric-container'><h3>Status</h3>", unsafe_allow_html=True)
    status_display = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# Camera setup
camera = cv2.VideoCapture(0)
closed_start = None
alerted = False

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("⚠️ Cannot access camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
            EAR = (left_EAR + right_EAR) / 2.0

            # Display EAR
            ear_display.metric(label="", value=f"{EAR:.3f}")

            # Drowsiness detection
            if EAR < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = time.time()
                elif time.time() - closed_start >= EYE_CLOSED_SECONDS and not alerted:
                    status_display.markdown("<p style='color:#ff4b4b; font-size:20px;'>🚨 Drowsy Detected!</p>", unsafe_allow_html=True)
                    play_beep()
                    alerted = True
            else:
                closed_start = None
                alerted = False
                status_display.markdown("<p style='color:#00ff88; font-size:20px;'>✅ Awake</p>", unsafe_allow_html=True)

    FRAME_WINDOW.image(frame_rgb)

camera.release()
st.success("🛑 Detection stopped.")
