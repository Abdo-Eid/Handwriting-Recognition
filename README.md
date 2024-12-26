# Handwriting-Recognition
try to learn and get good results for HWR

![](https://github.com/Abdo-Eid/Handwriting-Recognition/blob/main/Demo.gif)

# Model
- Using the `EMNIST` letter split with augmentation I got a good `Test Accuracy: 93.35%` but on separate letters
- you can see the training code in the [Notebook](emnist-hwr.ipynb)
- then saved the model as `ONNX` for speeding up the GUI and make the application lightweight

# Word Recognition
- I took the approach of detecting the boundaries of each letter in the image, making a batch to model, getting predictions, and combining the output (for more flexibility).
- One problem for letters that contain multiple parts like **i** I merge the 2 bbox (if there is overlap horizontally and close enough vertically).

# GUI
- I made a simple GUI that has a canvas to draw on the word then predict the word and show the bboxes of the letters

# Installing
- first clone the repo `git clone https://github.com/Abdo-Eid/Handwriting-Recognition.git`
- then change the directory `cd Handwriting-Recognition`
- then install the requirements `pip install -r requirements.txt`