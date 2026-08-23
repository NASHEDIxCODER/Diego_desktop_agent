"""
Encode face images and save to local pickle file.

Firebase upload is OPTIONAL — only performed if serviceAccountKey.json exists.
Local authentication does NOT require Firebase.
"""

import os
import pickle
from pathlib import Path

import cv2 as cv
import face_recognition


def encode_and_upload_faces():
    """Encode all face images in ./images/ and save to Known_encodings.p.
    
    Firebase upload is performed only if serviceAccountKey.json exists.
    """
    # importing the users images
    folderPath = 'images'
    pathlist = os.listdir(folderPath)

    imglist = []
    userName = []
    for path in pathlist:
        imglist.append(cv.imread(os.path.join(folderPath, path)))
        userName.append(os.path.splitext(path)[0])

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

    Known_encodings = FindEncodings(imglist)
    Known_EncodingWithName = [Known_encodings, userName]

    # Save locally (always)
    file = open("Known_encodings.p", 'wb')
    pickle.dump(Known_EncodingWithName, file)
    file.close()
    print(f"Saved {len(Known_encodings)} face encodings to Known_encodings.p")

    # Upload to Firebase (optional — only if serviceAccountKey.json exists)
    service_account_path = Path("serviceAccountKey.json")
    if service_account_path.exists():
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            options = {
                'databaseURL': "https://Diego-assit-default-rtdb.firebaseio.com/",
                'storageBucket': "gs://Diego-assit.appspot.com"
            }
            cred = credentials.Certificate(str(service_account_path))
            firebase_admin.initialize_app(cred, name="Diego assist", options=options)
            db = firestore.client(firebase_admin.get_app("Diego assist"))

            faces_ref = db.collection('faces')
            for i, encoding in enumerate(Known_encodings):
                face_data = {
                    'encoding': encoding.tolist(),
                    'name': userName[i]
                }
                faces_ref.document(f'face_{userName[i]}').set(face_data)

            print("Face encodings uploaded to Firebase successfully.")
        except Exception as e:
            print(f"Firebase upload skipped (optional): {e}")
    else:
        print("serviceAccountKey.json not found — Firebase upload skipped (optional)")


if __name__ == "__main__":
    encode_and_upload_faces()
