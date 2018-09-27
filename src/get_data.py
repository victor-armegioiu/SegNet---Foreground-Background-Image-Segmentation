import os
import numpy as np 
from skimage.io import imread, imshow, imsave
from skimage.transform import resize

class DataBatch(object):
	def __init__(self, train=None, test=None, image_dim=224):
		self.last_batch_idx = 0

		self.data_path = '../data'
		self.data_path += '/Test' if test else '/Train'

		self.X_names = sorted(os.listdir(self.data_path + '/Imgs'))
		self.y_names = sorted(os.listdir(self.data_path + '/GTs'))

		self.image_dim = image_dim

	def get_next_batch(self, batch_size):
		X =	np.zeros((batch_size, self.image_dim, self.image_dim, 3))
		y = np.zeros((batch_size, self.image_dim, self.image_dim))

		curr_sample = 0
		limit = min(self.last_batch_idx + batch_size, len(self.X_names))

		for i in range(self.last_batch_idx, self.last_batch_idx + batch_size):
			X_image = imread(self.data_path + '/Imgs/' + self.X_names[i])
			y_image = imread(self.data_path + '/GTs/' + self.y_names[i])

			X_image = resize(X_image, (self.image_dim, self.image_dim, 3))
			y_image = resize(y_image, (self.image_dim, self.image_dim))

			X[curr_sample] = X_image
			y[curr_sample] = y_image

		self.last_batch_idx = limit

		X = X.transpose(0, 3, 1, 2)
		return X, y

	def batch_idx(self):
		return self.last_batch_idx


if __name__ == '__main__':
	data_batch = DataBatch(test=True)
	X_batch, y_batch = data_batch.get_next_batch(batch_size=50)