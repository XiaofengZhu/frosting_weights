from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img = Image.open('./NIKE.png')
img = np.array(img)
plt.imshow(img)
plt.show()

HEIGHT, WIDTH = img.shape[1], img.shape[0]
#print(HEIGHT, WIDTH)
# Shifting Left
for i in range(HEIGHT, 1, -1):
  for j in range(WIDTH):
     if (i < HEIGHT-2):
       img[j][i] = img[j][i-2]
     elif (i < HEIGHT-1):
       img[j][i] = 255
plt.imshow(img)
plt.show()