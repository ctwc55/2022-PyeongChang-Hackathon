from flask import Flask, render_template, Response
from flask_cors import CORS
#from yolov5 import fast_detect
import cv2
import subprocess
import numpy as np

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
           
            #fast_detect.run(frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def mainPage():
    return render_template('index.html')

@app.route('/webcam.html')
def videoPage():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<name>')
def otherPages(name):
    return render_template(name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')