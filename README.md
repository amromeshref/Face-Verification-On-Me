# Face_Verification_On_Me

#### Video Demo: https://youtu.be/98Y7Qg1k6lk 

#### Description:

- I created a face verification app on me.
- The app takes an input image from a webcam then it verifies this image.
  
### table of contents
- [Data Collection](#Data-Collection)
- [How to Install](#how-to-install)
- [Usage](#usage)

  <div align="center">
<img src= "app.png" style="width:600px;height:600;">
</div>



## Data Collection

- I used the siamese neural network to train my model.
- To train a Siamese neural network, you need a dataset of pairs of inputs, where each pair is labeled as either similar or dissimilar.
- I collected the negative images (Disimilar Images) from  http://vis-www.cs.umass.edu/lfw/
- I collected anchor and positive images (Similar Images) from my webcam

## How to Install
1. Create a directory in your device
2. Download the siamese model architecture.
   - File name: siamese model.zip
   - Download Link: https://www.kaggle.com/datasets/amromeshref/face-verification-on-me
   - Make sure to uncompress the file
3. Download these files from this repository:
   - setup.py
   - layers.py
   - requirements.txt
   - verification images
4.  Type the following command to install the requirements using pip.
    ```bash
    pip install -r requirements.txt
    ```
5.  Type the following command to run the app.
    ```bash
    python3 setup.py
    ```  
