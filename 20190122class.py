# Numpy Random
import numpy as np
from numpy.core._multiarray_umath import ndarray

"""
## 랜덤 ##

data = { i: np.random.randn() for i in range(7)}
print(data)

from numpy.random import randn
data = {i: randn() for i in range(7)}
print(data)
"""

"""
## 배열연산 ##

x = np.array([[1,2],[3,4]], dtype = np.float64)
y = np.array([[5,6],[7,8]], dtype = np.float64)

print("\n+")
print(x+y,"\n")
print(np.add(x,y))

print("\n-")
print(x-y,"\n")
print(np.subtract(x,y))

print("\n*")
print(x*y,"\n")
print(np.multiply(x,y))

print("\n/")
print(x/y,"\n")
print(np.divide(x,y))
"""

"""
## Dot 함수 ##

dot_x = np.array([[1,2],[3,4]])
dot_y = np.array([[5,6],[7,8]])

dot_v = np.array([9,10])
dot_w = np.array = np.array([11,12])

print(dot_v.dot(dot_w))
print(np.dot(dot_v, dot_w))

print(dot_x.dot(dot_v))
print(np.dot(dot_x,dot_y))

print(dot_x.dot(dot_y))
print(np.dot(dot_x,dot_y))
"""

"""
##  Transpose   ##
t_x = np.array([[1,2],[3,4]])
print(t_x)
print(t_x.T)

t_v = np.array([1,2,3])
print(t_v)
print(t_v.T)

t_y = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
print(t_y)
print(t_y.T)
"""

"""
##  SciPy   ##
from scipy.misc import imread, imsave, imresize
img = imread('dog.jpg')
print(img.dtype, img.shape)
img_tinted = img * [1, 0.5, 0.5]
img_tinted = imresize(img_tinted, (300,300))
imsave('dog_tinted.jpg',img_tinted)
"""

"""
##  matplotlib  ##
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 3*np.pi , 0.1)
y = np.sin(x)
plt.plot(x,y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Sin')
plt.legend(['Sin'])
plt.show()
"""

"""
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
"""

"""
##  Image show  ##
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

img = imread('dog.jpg')
img_tinted = img * [0.5, 0.5, 1]

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted))
plt.show()
"""


"""
##  Folium  ##
import folium


m = folium.Map(location=[35.8915864, 128.86331619999999])
m = folium.Map(
    location=[35.8915864,128.86331619999999],
    zoom_start=12,
    tiles='평사'
)

tooltip = '평사'

folium.Marker([35.8915864, 128.86331619999999], popup='<i> 평사 </i>', tooltip=tooltip).add_to(m)

m

"""


