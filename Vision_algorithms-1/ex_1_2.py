"""Load an image and display it with a title and axis turned off."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

I = plt.imread('../mandrill.jpg')
#plt.figure(1)
fig,ax = plt.subplots(1) # swapped for plt.figure(1)
rect = Rectangle((50, 50), 50, 100, fill = False, ec = 'r'); # ec - edge colour
ax.add_patch(rect) # display of the rectangle
plt.imshow(I)
plt.title('Mandril')
plt.axis('off')
x = [100, 150, 200, 250]
y = [50, 100, 150, 200]
plt.plot(x, y, 'r.', markersize = 10)
plt.show()

plt.imsave('mandril.png', I)