from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
import face_recognition
from twilio.rest import Client

app = Flask(__name__)  # ✅ fixed typo (_name_ -> __name__)

DATA_DIR = "data"

def create_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# Function to extract face encodings
def extract_face_encodings(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings

# Function to send SMS using Twilio
def send_sms(message, recipient):
    account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
    auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
    twilio_number = '+16282127558'

    client = Client(account_sid, auth_token)

    client.messages.create(
        body=message,
        from_=twilio_number,
        to=recipient
    )

def detect_faces(frame):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file_name), "r") as file:
                details = file.readlines()
                face_encoding_str = details[4].split(":")[1].strip()
                face_encoding = np.fromstring(face_encoding_str, sep=',')
                known_face_encodings.append(face_encoding)
                known_face_names.append(details[0].split(":")[1].strip())

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            send_sms(f"The missing person {name} has been found!", '+916300938871')

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        location = request.form['location']
        phone_number = request.form['phone_number']
        image = request.files['image']

        image_filename = f"{name}.jpg"
        image_path = os.path.join(DATA_DIR, image_filename)
        image.save(image_path)

        face_locations, face_encodings = extract_face_encodings(image_path)

        if face_encodings:
            with open(os.path.join(DATA_DIR, f"{name}.txt"), "w") as file:
                file.write(f"Name: {name}\n")
                file.write(f"Location: {location}\n")
                file.write(f"Phone Number: {phone_number}\n")
                file.write(f"Image: {image_filename}\n")
                file.write(f"Face Encoding: {','.join(str(x) for x in face_encodings[0])}\n")

        return redirect(url_for('home'))

    return render_template('reg.html')

@app.route('/find_missing_person')
def find_missing_person():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    create_data_dir()
    app.run(debug=True)
