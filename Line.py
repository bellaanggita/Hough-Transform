######################
# Line Hough Transform
# ====================
#import package yang dibutuhkan
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm

# Mengkonstruksi gambar test
image = np.zeros((200, 200))
idx = np.arange(25, 175)
image[idx[::-1], idx] = 255
image[idx, idx] = 255
# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
# inisialisasi derajat garis sebesar 0.5 derajat
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
h, theta, d = hough_line(image, theta=tested_angles)
# Generating figure 1
#menampilkan gambar input yang sudah dibaca sebelumnya sebagai grayscale
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()
ax[0].imshow(image, cmap=cm.gray)
#beri judul gambar dengan 'Input image'
ax[0].set_title('Input image')
ax[0].set_axis_off()
#menampilkan gambar hasil hough transform
ax[1].imshow(np.log(1 + h),
 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
 cmap=cm.gray, aspect=1/1.5)
#beri judul gambar dengan 'hough transform'
ax[1].set_title('Hough transform')
#beri label sumbu x dengan 'angles (degrees)' untuk parameter angle dengan satuan derajatn
ax[1].set_xlabel('Angles (degrees)')
#beri label sumbu y dengan 'distance (pixels)' untuk parameter jarak dengan satuan pixel
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')
#menampilkan gambar hasil deteksi garis menggunakan hough transform
ax[2].imshow(image, cmap=cm.gray)
origin = np.array((0, image.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
 y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
 ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
#beri judul gambar dengan 'detected lines'
ax[2].set_title('Detected lines')
#menampilkan plot gambar
plt.tight_layout()
plt.show()

###############################
# Probabilistic Hough Transform
# =============================
#import probabilistic_hough_line dari package skimaga.transform untuk deteksi garis
from skimage.transform import probabilistic_hough_line
# Line finding using the Probabilistic Hough Transform
#mencari garis menggunakan probabilistic hough transform
#inisialisasi gambar dari data
image = data.camera()
#deteksi edge menggunakan canny detector
edges = canny(image, 2, 1, 25)
#deteksi point garis dengan apply hough transform
#kemudian gambarkan garis tersebut dengan nilai threshold 10, panjang 5 dan gap 3
lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
 line_gap=3)
# Generating figure 2 untuk menampilkan gambar hasil hough transform
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()
#menampilkan gambar input
ax[0].imshow(image, cmap=cm.gray)
#beri judul gambar dengan 'input image'
ax[0].set_title('Input image')
#menampilkan gambar edge hasil deteksi menggunakan canny
ax[1].imshow(edges, cmap=cm.gray)
#beri judul gambar dengan 'canny edges'
ax[1].set_title('Canny edges')
#menampilkan gambar hasil deteksi garis dengan menggunakan probabilistic hough transform
ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
#beri judul gambar dengan 'Probabilistic Hough'
ax[2].set_title('Probabilistic Hough')
for a in ax:
    a.set_axis_off()
plt.tight_layout()
#menampilkan plot
plt.show()
