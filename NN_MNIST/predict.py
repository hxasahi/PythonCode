import numpy as np
import math
import parameters
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(threshold=np.inf)

def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()

activation=[tanh,softmax]

parameters=parameters.parameters

def predict(img,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=activation[1](l1_in)
    return l1_out

img_gray = np.array(Image.open("./valid_dataset/0.bmp").convert("L"),"f")
img_gray = np.reshape(img_gray,(1,784))
print(predict(img_gray,parameters).argmax())
