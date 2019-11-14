## Code for paper: Improving the performance of Convolutional Neural Network for the Optic Disc segmentation in fundus images using Attention Gates and Conditional Random Fields

The repository contains code for running inference on the various models proposed in the paper. 
All the models can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/10yIz5iQ0yEDhlZS4fneT7pPOjcK5a3xo?usp=sharing).

Instructions to run the code:

```
python3 predict.py <MODEL> <INPUT FOLDER> <DST FOLDER>
```

Please ensure that the images in the input folder are named numerically as follows:

```
0001.jpg
0002.jpg
...
0099.jpg
...
```

We include a few sample images in the a folder called `sample_imgs` which follows the numerical 
naming scheme. 

We also include TensorFlow code of our proposed attention gates in `attn_gate.py` with docstring under the 
function explaining the arguments and return value. 

We currently only provide the trained models and inference code. 
For training code please email: [dheeraj98reddy@gmail.com](mailto:dheeraj98reddy).

The code was run and tested on Python3.6, and the dependencies are listed in `requirements.txt`.
