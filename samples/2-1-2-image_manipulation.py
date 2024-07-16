from matplotlib import image
from matplotlib import pyplot
import os
from PIL import Image
import numpy as np

# Read an image file
path = os.path.dirname(os.path.abspath(__file__))

filename = path + '\\' + 'lenna.bmp' 
filename1 = path + '\\' + 'india-flag-png-large.bmp'

print(filename)
print(filename1)

data = image.imread(filename)
data1 = Image.open(filename1)    

d = np.asarray(data1.resize((250,166)))

# Display image information
print('Image type is: ', type(data))
print('Image shape is: ', data.shape)
print('Image2 type is: ', type(data1))
print('Image2 shape is: ', d.shape)

# Add some color boundaries to modify an image array
plot_data = data.copy()

for width in range(250):
    for height in range(166):
        plot_data[height][512-250+width] = d[height][width] 

# Write the modified images
image.imsave(path+'/'+'lenna-mod.jpg', plot_data)

# use pyplot to plot the image
pyplot.imshow(plot_data)
pyplot.show()