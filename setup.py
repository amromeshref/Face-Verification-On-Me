# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


def collecting_verification_images():
        ver_images = []
        # Path for some anchor images will be used to verify the input images
        ver_path = "my data/verification"   
        for img_name in os.listdir(ver_path):
            path = ver_path + "/" + img_name
            image = cv2.imread(path)
            ver_images.append(image)
        ver_images = np.array(ver_images)

        # Normalizing the verification images
        ver_images = ver_images/ [255]

        # Resizing the verification images to (105,105,3)
        temp = []
        for i in range(ver_images.shape[0]):
            img = ver_images[i]
            img = tf.image.resize(img,(105,105))
            img = img.numpy()
            temp.append(img)
        temp = np.array(temp)
        temp = temp.reshape((ver_images.shape[0],1,105,105,3))
        ver_images = temp

        return ver_images

ver_images = collecting_verification_images()



# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('my data/siamese_model.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, img):
        # Normalizing the image
        img = img/255
        # resizing the image to (105, 105, 3)
        img = tf.image.resize(img,(105,105))
        img = img.numpy()
        img = img.reshape((1,105,105,3))
        
        return img

    def verify(self, *args):

        inp_img_path = "my data/input_image/input_image.jpg"

        # Getting the input image
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250, :]
        cv2.imwrite(inp_img_path, frame)

        # Preprocessing the input image
        input_image = cv2.imread(inp_img_path)
        input_image = self.preprocess(input_image)

        # Getting the verification images

        result = []
        for ver in ver_images:
            y_hat = self.model.predict([ver,input_image])
            if(y_hat[0,0] >= 0.9):
                y_hat[0,0] = 1
            else:
                y_hat[0,0] = 0
            result.append(y_hat[0,0])
        
        m = ver_images.shape[0]
        result = np.array(result)
        accuracy = 100*(np.sum(result == 1)/m)
        
        self.verification_label.text = 'Verified, Welcome Amro!' if accuracy > 60 else 'Unverified'
        print("Accuracy =", accuracy)

if __name__ == '__main__':
    CamApp().run()