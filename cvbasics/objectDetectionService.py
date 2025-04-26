from flask import Flask
from flask import request
import os
import cv2 as cv
import numpy as np

from GraphicDataProcessing import ObjectDetection


app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

@app.route('/detect', methods=['POST'])
def detection():
    args = request.args
    
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)

    img = cv.imdecode(np.frombuffer(imagefile.read(), np.uint8), cv.IMREAD_COLOR)

    obj_detection = ObjectDetection(img)

    findings = obj_detection.detect()

    findings_string = str(findings)
    
    return findings_string 
    


if __name__ == "__main__":
    flaskPort = 8788
    
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

    