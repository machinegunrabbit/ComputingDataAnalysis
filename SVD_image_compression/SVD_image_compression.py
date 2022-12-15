
# # SVD-based (image) compression

# ## Setup

# Step 1 Load an image:
# Step 2: Download an image and represent it by a Numpy matrix, `img`.
# Step 3: Convert image to grayscale 2-D array and print its dimensions and size (in pixels).

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
from PIL import Image

def im2gnp(image):
    """Converts a PIL image into an image stored as a 2-D Numpy array in grayscale."""
    return np.array(image.convert ('L'))

def gnp2im(image_np):
    """Converts an image stored as a 2-D grayscale Numpy array into a PIL image."""
    return Image.fromarray(image_np.astype(np.uint8), mode='L')

def imshow_gray(im, ax=None):
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    ax.imshow(im,
              interpolation='nearest',
              cmap=plt.get_cmap('gray'))

import requests
import os
import hashlib
import io

def on_vocareum():
    return os.path.exists('.voc')

def download(file, local_dir="", url_base=None, checksum=None):
    local_file = "{}{}".format(local_dir, file)
    print (local_file)
    if not os.path.exists(local_file):
        if url_base is None:
            url_base = "https://cse6040.gatech.edu/datasets/"
        url = "{}{}".format(url_base, file)
        print("Downloading: {} ...".format(url))
        r = requests.get(url)
        with open(local_file, 'wb') as f:
            f.write(r.content)

    if checksum is not None:
        with io.open(local_file, 'rb') as f:
            body = f.read()
            body_checksum = hashlib.md5(body).hexdigest()
            assert body_checksum == checksum,                 "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_file,
                                                                                           body_checksum,
                                                                                           checksum)
    print("'{}' is ready!".format(file))

URL_BASE = "https://cse6040.gatech.edu/datasets/tech-tower/"
DATA_PATH = "./resource/asnlib/publicdata/" if on_vocareum() else ""
datasets = {'tt1.jpg': '380479dfdab7cdc100f978b0e00ad814'}

for filename, checksum in datasets.items():
    download(filename, local_dir=DATA_PATH, url_base=URL_BASE, checksum=checksum)

print("\n(All data appears to be ready.)")

pic_raw = Image.open('{}tt1.jpg'.format(DATA_PATH), 'r')
pic = im2gnp(pic_raw)
imshow_gray(pic)

# Step 4: Given a 2-D grayscale Numpy array, returns the total number of pixels
def sizeof_image(I):
    assert type(I) is np.ndarray
    assert len(I.shape) == 2
    m,d = I.shape
    return m*d
pic_pixels = sizeof_image (pic)
print ("The image uses about {:.1f} megapixels.".format (1e-6 * pic_pixels))

# Step 5: Given an image I matrix, compute its U, Sigma, VkT at its K-trucated place, return three matrices as a tuple
def compress_image(I, k):
    u, sigma, v = np.linalg.svd(I, full_matrices=False)
    np.sort(sigma)
    U = u[:,0:k]
    VkT = v[0:k,:]
    return (sigma,U,VkT)

k = 10
Sigma, Uk, VkT = compress_image(pic, k)
print(Sigma.shape)
print(Uk.shape)
print(VkT.shape)
assert Sigma.shape == (min(pic.shape),)
assert Uk.shape == (pic.shape[0], k)
assert VkT.shape == (k, pic.shape[1])
assert (Sigma[k:]**2).sum () <= 7e9

# Step 6: Given the SVD representation of the image matrix, calculate its size. Consider "equivalent pixel"
def sizeof_compressed_image(Sigma, Uk, VkT):
    A = Sigma[::-1][0:10]
    u1,u2 = Uk.shape
    v1,v2 = VkT.shape
    size = 8*(u1*u2+u2+v1*v2)
    return size

cmp_pixels = sizeof_compressed_image(Sigma, Uk, VkT)
print("Original image required ~ {:.1f} megapixels.".format (1e-6 * pic_pixels))
print("Compressed representation retaining k={} singular values is equivalent to ~ {:.1f} megapixels.".format (k, 1e-6 * cmp_pixels))
print("Thus, the compression ratio is {:.1f}x.".format (pic_pixels / cmp_pixels))

# Step 7: Calculate compression error at K-rank tructed place
def compression_error (Sigma, k):
    """
    Given the singular values of a matrix, return the
    relative reconstruction error.
    """
    sorted_desc = -np.sort(-Sigma)
    f_norm = np.linalg.norm(sorted_desc[k:])
    sigma_norm = np.linalg.norm(Sigma)
    relative_err = f_norm/np.linalg.norm(Sigma)
    return relative_err
print(Sigma)
print(k)
err = compression_error(Sigma, k)
print ("Relative reconstruction (compression) error is ~ {:.1f}%.".format (1e2*err))

# Step 8: Reconstruct the SVD representation of image

def uncompress_image(Sigma, Uk, VkT):
    assert Uk.shape[1] == VkT.shape[0]
    sorted_sigma = -np.sort(-Sigma)
    dig_sig = np.diag(sorted_sigma[:Uk.shape[1]])
    reduced_image = Uk.dot(dig_sig.dot(VkT))
    return reduced_image

pic_lossy = uncompress_image(Sigma, Uk, VkT)
f, ax = plt.subplots(1, 2, figsize=(15, 30))
imshow_gray(pic, ax[0]) ##function to visualize 2-D Numpy arrary representation of grayscale
imshow_gray(pic_lossy, ax[1])
abs_err = np.linalg.norm(pic - pic_lossy, ord='fro')
rel_err = abs_err / np.linalg.norm(pic, ord='fro')
print("Measured relative error is ~ {:.1f}%.".format(1e2 * rel_err))

# Step 9:
def find_rank(rel_err_target, Sigma):
    curr_compress_err = 0.99
    for k in range(Sigma.shape[0]):
        old_curr_compress_err = 0 + curr_compress_err
        if old_curr_compress_err >= rel_err_target:
            curr_compress_err = compression_error(Sigma,k)
            result = k
        else:
            break
    return result

rel_err_target = 0.15
k_target = find_rank(rel_err_target, Sigma)
print("Relative error target:", rel_err_target)
print("Suggested value of k:", k_target)
print("Compressing...")
Sigma_target, Uk_target, VkT_target = compress_image(pic, k_target)
target_pixels = sizeof_compressed_image(Sigma_target,
                                        Uk_target,
                                        VkT_target)
target_ratio = pic_pixels / target_pixels
print("Estimated compression ratio: {:.1f}x".format(target_ratio))
pic_target = uncompress_image(Sigma_target, Uk_target, VkT_target)
f, ax = plt.subplots(1, 2, figsize=(15, 30))
imshow_gray(pic, ax[0])
imshow_gray(pic_target, ax[1])
assert compression_error(Sigma, k_target) <= rel_err_target
assert compression_error(Sigma, k_target-1) > rel_err_target
