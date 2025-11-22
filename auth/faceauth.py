import os
import pickle

import cv2 as cv
import face_recognition
import firebase_admin
import numpy as np
import pyttsx3
import speech_recognition as sr
from firebase_admin import credentials, firestore

from auth.encode import encode_and_upload_faces


# --------- CONFIG ---------
CAM_INDEX = 0  # use 0 on most laptops; 1 was failing earlier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")
IMAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "images")  # ../images


# --------- TTS ----------
engine = pyttsx3.init()  # let pyttsx3 choose (espeak on Linux)
voices = engine.getProperty("voices")
if voices:
    engine.setProperty("voice", voices[0].id)


def speak(audio: str):
    engine.say(audio)
    engine.runAndWait()


# --------- SR Helper ----------
def listen_for_command() -> str:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        r.adjust_for_ambient_noise(source, duration=0.7)
        audio = r.listen(source)

    try:
        print("Recognizing...")
        command = r.recognize_google(audio, language="en-in")
        print(f"User said: {command}\n")
        return command.lower()
    except Exception as e:
        print("SR error:", e)
        speak("Sorry, I couldn't understand that.")
        return ""


# --------- Firebase init ----------
def get_firestore():
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        print(f"[FATAL] serviceAccountKey.json not found at: {SERVICE_ACCOUNT_PATH}")
        speak("Firebase credentials file is missing.")
        return None

    options = {
        "databaseURL": "https://leo-assit-default-rtdb.firebaseio.com/",
        "storageBucket": "gs://leo-assit.appspot.com",
    }

    try:
        app = firebase_admin.get_app("leo assist")
    except ValueError:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        app = firebase_admin.initialize_app(cred, name="leo assist", options=options)

    return firestore.client(app)


# --------- New face enrollment ----------
def Unknown_Face():
    """Enroll a new face: ask name, capture photo, save to images/, re-encode."""
    print("[INFO] Starting unknown face enrollment...")
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("[FATAL] Could not open camera for Unknown_Face.")
        speak("I cannot access the camera right now.")
        return False

    os.makedirs(IMAGES_DIR, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Empty frame in Unknown_Face, skipping.")
            continue

        small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

        faces = face_recognition.face_encodings(rgb_small_frame)
        cv.imshow("New face enrollment", small_frame)

        if faces:
            print("Face detected for enrollment.")
            speak("Tell me your name please.")
            name = listen_for_command().strip()

            if name:
                # Save original full-size frame to images/<name>.jpg
                img_path = os.path.join(IMAGES_DIR, f"{name}.jpg")
                success = cv.imwrite(img_path, frame)
                if success:
                    print(f"[INFO] Saved face image as {img_path}")
                    speak(f"Saving your face as {name}")
                    # Re-encode all faces (existing + new)
                    encode_and_upload_faces()
                    cap.release()
                    cv.destroyAllWindows()
                    return True
                else:
                    print("[ERROR] Failed to save image.")
                    speak("Something went wrong while saving your face.")
            else:
                print("[WARN] No name captured, retrying enrollment...")
                speak("I didn't get your name, please try again.")
                # continue loop and try again
        else:
            print("No face detected in current frame.")

        if cv.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User chose to exit Unknown_Face.")
            break

    cap.release()
    cv.destroyAllWindows()
    return False


# --------- Face recognition ----------
def recognize_faces():
    db = get_firestore()
    if db is None:
        return None

    # Load encodings
    enc_path = os.path.join(os.path.dirname(BASE_DIR), "auth/Known_encodings.p")
    if not os.path.exists(enc_path):
        print(f"[FATAL] Known_encodings.p not found at: {enc_path}")
        speak("Face encodings file is missing. Please run encoding first.")
        return None

    print("loading encode file")
    with open(enc_path, "rb") as f:
        Known_EncodingWithName = pickle.load(f)
    Known_encodings, userName = Known_EncodingWithName
    print("encode file loaded")

    cam = cv.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        print("[FATAL] Could not open camera for recognize_faces.")
        speak("I cannot access the camera right now.")
        return None

    Process_this_frame = True

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[WARN] Empty frame in recognize_faces, skipping.")
            continue

        if Process_this_frame:
            # resize the frame
            small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("Face recognition", frame)

            # convert BGR to RGB
            rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

            face_currentFrame = face_recognition.face_locations(rgb_small_frame)
            encodeCurrentFrame = face_recognition.face_encodings(
                rgb_small_frame, face_currentFrame
            )

            for encodeFace, faceLoc in zip(encodeCurrentFrame, face_currentFrame):
                if len(Known_encodings) == 0:
                    print("[WARN] No known encodings yet.")
                    matches = []
                    faceDis = []
                else:
                    matches = face_recognition.compare_faces(Known_encodings, encodeFace)
                    faceDis = face_recognition.face_distance(Known_encodings, encodeFace)

                print("matches:", matches)
                print("distances:", faceDis)

                if len(faceDis) > 0:
                    matchindex = np.argmin(faceDis)
                    if matches[matchindex]:
                        name = userName[matchindex]
                        print(f"[INFO] Recognized: {name}")
                        speak(name)
                        cam.release()
                        cv.destroyAllWindows()
                        return name
                    else:
                        print("[INFO] Unknown face encountered.")
                        speak("Unknown face")
                        enrolled = Unknown_Face()
                        if enrolled:
                            # reload encodings after enrollment
                            print("[INFO] Reloading encodings after enrollment...")
                            with open(enc_path, "rb") as f:
                                Known_EncodingWithName = pickle.load(f)
                            Known_encodings, userName = Known_EncodingWithName
                        # continue loop to try again

        if cv.waitKey(1) & 0xFF == ord("q"):
            print("you chose to exit.")
            break

    cam.release()
    cv.destroyAllWindows()
    return None


if __name__ == "__main__":
    recognize_faces()
