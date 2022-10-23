from flask import Flask, render_template, Response
from flask_cors import CORS
#from yolov5 import fast_detect
import cv2
import subprocess
import numpy as np
import asyncio
import os

app = Flask(__name__)
CORS(app)

def gen_frames():
    #os.system("python yolov5/fast_detect")    
    
    while True:
        #img = cv2.imread('static/images/img.jpg', cv2.IMREAD_COLOR)
        camera = cv2.VideoCapture('./static/images/img.jpg')
        success, img = camera.read()

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_frames2():
    #os.system("python yolov5/fast_detect")    
    
    while True:
        #img = cv2.imread('static/images/img.jpg', cv2.IMREAD_COLOR)
        camera = cv2.VideoCapture('./static/images/img2.jpg')
        success, img = camera.read()

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    '''
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            print(type(frame))
            print(frame.shape)
             
            f = next(fast_detect.run()) 
            
            ret, buffer = cv2.imencode('.jpg', f)
            frame = buffer.tobytes()
            


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    '''

@app.route('/')
def mainPage():
    return render_template('index.html')

@app.route('/webcam.html')
def videoPage():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<name>')
def otherPages(name):
    return render_template(name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')