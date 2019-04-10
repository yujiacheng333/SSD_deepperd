import numpy as np
import PIL.Image as img
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width,3)).astype(np.uint8)
img = img.open("./test/4.jpg")
img = load_image_into_numpy_array(img)
print(np.sum(img,axis=2).shape)
plt.imshow(np.sum(img,axis=2))
plt.show()