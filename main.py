import HoG_Calculate
import cv2
import kmeans as km

'''
cell_size=8
block_size=2
bins=9
block_stride=1

image = cv2.imread("FELV-cat.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_resized = cv2.resize(gray_image, (128,64))


normalized_block, hog_cells = HoG_Calculate.compute_hog(image_resized, cell_size, block_size, block_stride, bins)
'''

k = 10
km = km.Kmeans(500, k)
km.update_clusters()
km.compare_seeds(3)

