import rasterio
import os
import numpy as np
import pdb


split_path = 'dfc-train.csv'

label_fnames = []

with open(split_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        fname = line.strip()
        label_fnames.append(fname)

print(len(label_fnames))

# band_sum = np.zeros(13, dtype=np.float64)
# band_squared_sum = np.zeros(13, dtype=np.float64)
# pixel_counts = np.zeros(13, dtype=np.int64)

# for label_fname in label_fnames:
#     img_fname = label_fname.replace('dfc','s2')
#     img_path = os.path.join('s2', img_fname)

#     with rasterio.open(img_path) as src:
#         img = (src.read() / 10000.0).astype('float32')

#     #pdb.set_trace()
#     for band in range(img.shape[0]):
#         band_data = img[band]  # Data for the current band
#         band_sum[band] += band_data.sum()
#         band_squared_sum[band] += np.square(band_data).sum()
#         pixel_counts[band] += band_data.size

# #pdb.set_trace()

# # Compute the mean for each band
# band_mean = band_sum / pixel_counts

# # Compute the variance and then the standard deviation
# band_variance = band_squared_sum / pixel_counts - np.square(band_mean)
# band_std = np.sqrt(band_variance)

# print("Band-wise Mean:", band_mean)
# print("Band-wise Std:", band_std)


cls_sum = np.zeros(11, dtype=np.float64)
pix_count = 0

for label_fname in label_fnames:
    #img_fname = label_fname.replace('dfc','s2')
    label_path = os.path.join('dfc', label_fname)

    with rasterio.open(label_path) as src:
        label = src.read(1)
        for i in range(11):
            label_bool = np.zeros(label.shape)
            label_bool[label==i] = 1
            cls_pix = label_bool.sum()
            cls_sum[i] += cls_pix
        pix_count += label.size

cls_pct = cls_sum / pix_count

print('label distribution:', cls_pct)