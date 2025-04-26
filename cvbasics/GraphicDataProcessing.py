import cv2 as cv
from scipy import ndimage
from skimage.util import random_noise
import numpy as np

class ObjectDetection:
    def __init__(self, img):
        self.img = img

    def shape(self):
        
        if self.img.shape is not None:
            shape = self.img.shape
            print("Original Shape:", shape)
        else:
            print("Failed to read the image.")
            
    def resize(self, new_size):
        self.img = cv.resize(self.img, new_size)
        
        if isinstance(self.img, type(None)):
            print("Failed to resize the image. Image not resized.")
        else:
            
            print("Resized Shape:", self.img.shape)
    
    def rotate(self, angle): # improve later; include anti-clockwise option
        
        self.img = ndimage.rotate(self.img, angle)
        
        if isinstance(self.img, type(None)):
            print("Failed to rotate the image. Image not rotated.")
        else:
            
            print(f"Image rotated {angle} degrees") # this has to be changed later
    
    def noise_salt_pepper(self, amount):
        
        
        if isinstance(self.img, type(None)):
            print("Failed to add noise to the image.")
        else:
            noisy_img = random_noise(self.img, mode='s&p', amount=amount)
            noisy_img = np.array(255 * noisy_img, dtype='uint8')
            print(f"Salt & Pepper {amount} noise added")
    
    def noise_gaussian(self, var): # variance
        
        if isinstance(self.img, type(None)):
            print("Failed to add noise to the image.")
        else:
            gaussy_img = random_noise(self.img, mode='gaussian', var=var)
            gaussy_img = np.array(255 * gaussy_img, dtype='uint8')
            print(f"Gaussian noise with variance {var} added")
    
    def noise_speckle(self,var): # amount = 0.02, 0.05, 0.1, 0.2 etc
        
        if isinstance(self.img, type(None)):
            print("Failed to add noise to the image.")
        else:
            speckled_img = random_noise(self.img, mode='speckle', var=var)
            speckled_img = np.array(255 * speckled_img, dtype='uint8')
            print(f"Speckled noise with variance {var} added")
    
    def change_loop(self,new_size,new_angle,noise): # newsize is a tuple
        
        if isinstance(self.img, type(None)):
            print("Failed to modify the image.")
        else:
            original_img = self.img.copy() # for reseting
            self.img = cv.resize(original_img, new_size)
            self.img = ndimage.rotate(self.img, new_angle)
            
            if noise == 'Gaussian':               
                self.img =self.noise_gaussian(0.5)
            elif noise =='Salt & Pepper':
                self.img = self.noise_salt_pepper(0.5)
            elif noise == 'Speckle':
                self.img = self.noise_speckle(0.2)
            print("Image is rotated, resized and {noise} added.")
        
    def detect(self):                
        
        net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
        ln = net.getLayerNames()
        classes = []
        with open("coco.names", 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        layer_name = net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
                     

        img = cv.resize(self.img, (416, 416))  # Add code to resize
        height, width, channel = img.shape
        #print("Resized Shape: ", img.shape)

        # Define parameters
        scale_factor = 1/255.0  # Scale factor to normalize pixel values to [0, 1]
        size = (416, 416)       # Standard size for YOLO input
        mean = (0, 0, 0)        # Mean subtraction (zero mean)
        swapRB = True           # Swap Red and Blue channels

        # Create blob from image
        blob = cv.dnn.blobFromImage(img, scale_factor, size, mean, swapRB) # change

        #detect objects
        net.setInput(blob)
        outs = net.forward(output_layer)

        # Showing Information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
                    # Reactangle Cordinate
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                return label, confidences[i]
        

        


                
                
