import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PIL.Image as image
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width)).astype(np.uint8)


img = image.open("./lable_images/0000000593.png").resize((1282,276),image.ANTIALIAS)

img = load_image_into_numpy_array(img)
disp_from_depth = 0.54 * 718.856 / (img)
disp_from_depth[img < 0] = -1
plt.imshow(disp_from_depth)
plt.show()
plt.imshow(img)

img = np.sum(img,axis=2)
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(0, 255, 1)
Y = np.arange(0, 255, 1)
X, Y = np.meshgrid(X, Y)
plt.figure(figsize=(16,9))
ax.view_init(elev=50., azim=135)
ax.plot_surface(X, Y, img, rstride=10, cstride=10, cmap='rainbow')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
'''
#scatter
x = []
y = []
z = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        z.append(i)
        x.append(j)
        y.append(img[i,j])

ax = plt.figure().add_subplot(111, projection = '3d')
#ax.scatter(x, y, z, c='r', marker='^')
ax.plot_surface(x, y, z, rstride=10, cstride=10, cmap='rainbow')
# 设置坐标轴
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
'''