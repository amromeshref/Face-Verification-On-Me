# Face ID


#### Description:

- I created a face verification app on me.
- The app takes an input image from a webcam then it verifies this image.

<div align="center">
<img src= "app.png" style="width:600px;height:600;">
</div>

### Table of Contents
- [Data Collection](#Data-Collection)
- [How to Install](#how-to-install)



## Data Collection

- I used the siamese neural network to train my model.
    - Paper Link: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- To train a Siamese neural network, you need a dataset of pairs of inputs, where each pair is labeled as either similar or dissimilar.
- I collected the negative images (Disimilar Images) from  http://vis-www.cs.umass.edu/lfw/
- I collected anchor and positive images (Similar Images) from my webcam.

## How to Install
1. Create a directory in your device.
2. Download the siamese model architecture.
   - File name: siamese_model.rar
   - Download Link: https://www.kaggle.com/datasets/amromeshref/face-verification-on-me
   - Make sure to uncompress the file.
3. Download these files from this repository:
   - app.py
   - layers.py
   - requirements.txt
   - verification images
4.  Type the following command to install the requirements using pip:
    ```bash
    pip install -r requirements.txt
    ```
5.  Type the following command to run the app:
    ```bash
    python3 app.py
    ```  
