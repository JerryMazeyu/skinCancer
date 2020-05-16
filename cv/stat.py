import os
from ProcessImage import ProcessImage
import numpy as np
import matplotlib.pyplot as plt

prim = ProcessImage()

targetdir = '/Users/mazeyu/newEraFrom2020.5/skinCancer/data/train-2c/melanoma'
targetdir1 = '/Users/mazeyu/newEraFrom2020.5/skinCancer/data/train-2c/nevus'

mel_res = []
for imgpath in os.listdir(targetdir):
    imgpath = os.path.join(targetdir,imgpath)
    print(imgpath, "is processing")
    res = prim(imgpath)
    mel_res.append(res)



ne_res = []
for imgpath in os.listdir(targetdir1):
    imgpath = os.path.join(targetdir1,imgpath)
    res = prim(imgpath)
    ne_res.append(res)

fig, ax = plt.subplots()
ax.plot(np.arange(len(mel_res)), mel_res)
ax.plot(np.arange(len(mel_res)), ne_res[:len(mel_res)])
plt.show()


print(np.average(mel_res),np.median(mel_res), "-->", np.average(ne_res), np.median(ne_res))