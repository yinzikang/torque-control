import mujoco_py
import numpy as np
import math
from PIL import Image



row = 200
col = 200

# sin table
# shape = np.array([math.sin(x/row*2*np.pi) for x in np.arange(row)] * col).reshape(row, col)
# shape = (shape+1)*255.0/2

# triangle table
# shape = np.array([x/row for x in np.arange(row)] * col).reshape(row, col)
# shape = shape*255.0

# step table
shape = np.array([x*0+1 for x in np.arange(row)] * col).reshape(row, col)
shape = shape*255.0

print(shape)
outputImg = Image.fromarray(shape)
#"L"代表将图片转化为灰度图
outputImg = outputImg.convert('L')
outputImg.save('meshes/shape.png')
outputImg.show()

mjc_model = mujoco_py.load_model_from_path("table.xml")
sim = mujoco_py.MjSim(mjc_model)
viewer = mujoco_py.MjViewer(sim)
while True:
    viewer.render()

