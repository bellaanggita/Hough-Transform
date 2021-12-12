"""
========================================
Circular and Elliptical Hough Transforms
========================================
"""
#import package yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

# Load picture and detect edges
#load gambar input
image = img_as_ubyte(data.coins()[160:230, 70:270])
#deteksi edge menggunakan canny detector
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
# Detect two radii
# untuk mendeteksi lingkaran dalam gambar, set 2 radius
hough_radii = np.arange(20, 35, 2)
# apply hough transform pada gambar menggunakan fungsi hough_circle
hough_res = hough_circle(edges, hough_radii)
# Select the most prominent 3 circles
# pilih 3 kandidat circle yang paling menonjol
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
 total_num_peaks=3)
# gambar lingkaran yang dideteksi
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
#convert ke grayscale
image = color.gray2rgb(image)
#set parameter yang digunakan
for center_y, center_x, radius in zip(cy, cx, radii):
 #atur parameter apa saja yang digunakan untuk membuat circle seperti koordinat lingkaran dan radius
 circy, circx = circle_perimeter(center_y, center_x, radius,
 shape=image.shape)
 #set besar nilai koordinat lingkaran dan radius
 image[circy, circx] = (220, 20, 20)
# tampilkan plot gambar
ax.imshow(image, cmap=plt.cm.gray)
plt.show()


######################################################################
# Ellipse detection
# =================
######################################################################
#import package yang dibutuhkan
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
# load gambar input
image_rgb = data.coffee()[0:220, 160:420]
#convert gambar ke grauscale
image_gray = color.rgb2gray(image_rgb)
#deteksi tepi gambar menggunakan canny detector
edges = canny(image_gray, sigma=2.0,
#tetapkan nilai untuk nilai maksimum dan minimum radius
 low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators

# Inisialisasi akumulator
# apply hough transform pada gambar menggunakan fungsi hough_ellipse
# set besar nilai parameter seperti threshold, nilai maksimum dan minimum jari-jari
result = hough_ellipse(edges, accuracy=20, threshold=250,
 min_size=100, max_size=120)
result.sort(order='accumulator')

# estimasi parameter untuk ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]
# gambar ellipse berdasarkan original picture
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
#gambar tepian(edge) ellipse (berwarna putih) dan hasil ellipse (berwarna merah)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)
fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
 sharex=True, sharey=True)
#beri judul gambar dengan 'Original picture'
ax1.set_title('Original picture')
ax1.imshow(image_rgb)
#beri judul gambar dengan 'Edge (white) and result (red)'
ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)
#tampilkan plot gambar
plt.show()
