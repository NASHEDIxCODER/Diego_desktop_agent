import os
import pickle

import cv2 as cv
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore


# from main import speak


def encode_and_upload_faces():
    options = {
        'databaseURL': "https://leo-assit-default-rtdb.firebaseio.com/",
        'storageBucket': "gs://leo-assit.appspot.com"
    }

    cred = credentials.Certificate("serviceAccountKey.json")
    # Initialize or get the Firebase app

    firebase_admin.initialize_app(cred, name="leo assist", options=options)

    # Initialize Firestore database
    db = firestore.client(firebase_admin.get_app("leo assist"))

    # importing the users images
    folderPath = 'images'
    pathlist = os.listdir(folderPath)

    imglist = []
    userName = []
    for path in pathlist:
        imglist.append(cv.imread(os.path.join(folderPath, path)))
        userName.append(os.path.splitext(path)[0])
        # userName.append(os.path.basename(path)[0])

    def FindEncodings(imagelist):
        encodeList = []
        for img in imagelist:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            faces = face_recognition.face_encodings(img)
            if faces:
                encode = faces[0]
                encodeList.append(encode)
            else:
                print(f"No face detected in image: {img}")
        return encodeList

    def upload_face_encodings_to_firebase(encodings, names):
        # ❌ faces_ref = db.collections('faces')
        # generator ^

        # ✅ this is a CollectionReference
        faces_ref = db.collection('faces')

        for i, encoding in enumerate(encodings):
            face_data = {
                'encoding': encoding.tolist(),  # numpy array -> list
                'name': names[i]
            }
            faces_ref.document(f'face_{names[i]}').set(face_data)

        print("Face encodings uploaded to Firebase successfully.")

    # speak("encoding started")
    Known_encodings = FindEncodings(imglist)
    Known_EncodingWithName = [Known_encodings, userName]
    # speak("encoding complete")
    # speak("uploading faces in database")
    upload_face_encodings_to_firebase(Known_encodings, userName)
    file = open("Known_encodings.p", 'wb')
    pickle.dump(Known_EncodingWithName, file)
    file.close()
    print("file close")
    # for name in userName:
    #     speak(name)

    # speak("faces uploaded successfully")


if __name__ == "__main__":
    encode_and_upload_faces()
