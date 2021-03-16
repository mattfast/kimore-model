import os
import scipy.signal
import scipy.fftpack
import numpy as np 
from load_kimore import kinect_joints

# convert all joint position/orientation and timestamp data to np arrays
def to_np_arrays(exercise_data):
  
  for joint in kinect_joints:
    exercise_data[joint + "-p"] = np.array(exercise_data[joint + "-p"]).astype(np.float)
    exercise_data[joint + "-o"] = np.array(exercise_data[joint + "-o"]).astype(np.float)

  exercise_data["Timestamps"] = np.array(exercise_data["Timestamps"])
  
  return exercise_data

def to_lists(exercise_data):

  for joint in kinect_joints:
	  exercise_data[joint + "-p"] = exercise_data[joint + "-p"].tolist()
	  exercise_data[joint + "-o"] = exercise_data[joint + "-o"].tolist()

  exercise_data["Timestamps"] = exercise_data["Timestamps"]

  return exercise_data


def convolve(signal, kernel):

	convolved_signal = []
	for i in range(len(signal)):
		sum = 0
		for j in range(len(kernel)):
			if i+j >= 0 and i+j < len(signal):
				sum += signal[i+j] * kernel[j]

		convolved_signal.append(sum)

	return np.array(convolved_signal)


# average joint position data along time dimension (with 1D convolution)
def averaging_filter(exercise_data, filter_size=5):

  kernel = [1 / filter_size for i in range(filter_size)]

  for joint in kinect_joints:
	  x_dim = exercise_data[joint + "-p"][:,0]
	  y_dim = exercise_data[joint + "-p"][:,1]
	  z_dim = exercise_data[joint + "-p"][:,2]

	  exercise_data[joint + "-p"][:,0] = convolve(x_dim, kernel)
	  exercise_data[joint + "-p"][:,1] = convolve(y_dim, kernel)
	  exercise_data[joint + "-p"][:,2] = convolve(z_dim, kernel)

  return exercise_data


# center wcs at spine base
def center_wcs(exercise_data):

  for joint in kinect_joints:
	  exercise_data[joint + "-p"] -= exercise_data["spinebase-p"]

  return exercise_data


# crops joint position data
# args:
# 	fixed - if True, crops data to a fixed length;
# 	   if False, crops data according to ratio r
# 	fixed_len - if fixed=True, crops data to fixed_len, 
# 	   using the middle portion of the data
# 	
def crop(exercise_data, fixed=False, fixed_len=100, r=0.1):

  if fixed:
	  for joint in kinect_joints:
		  len_data = len(exercise_data[joint + "-p"])
		  start = int((len_data / 2) - (fixed_len / 2))
		  end = int(start + fixed_len)

		  exercise_data[joint + "-p"] = exercise_data[joint + "-p"][start:end]
  else:
	  for joint in kinect_joints:
		  len_data = len(exercise_data[joint + "-p"])
		  start = int(r * len_data)
		  end = int((1 - r) * len_data)

		  exercise_data[joint + "-p"] = exercise_data[joint + "-p"][start:end]

  return exercise_data



# computes dft for every joint position
# args:
# 	k - num of frequencies returned for each joint
# returns dict with:
# 	keys - joint name
# 	vals - computed frequencies for the input joint position data
def dct(exercise_data, k=15):

	dct_dict = {}
	for joint in kinect_joints:
		x_pos = exercise_data[joint + "-p"][:,0]
		y_pos = exercise_data[joint + "-p"][:,1]
		z_pos = exercise_data[joint + "-p"][:,2]

		x_dct = scipy.fftpack.dct(x_pos)
		y_dct = scipy.fftpack.dct(y_pos)
		z_dct = scipy.fftpack.dct(z_pos)

		dct_vect = np.concatenate((x_dct[:k], y_dct[:k], z_dct[:k]))

		dct_dict[joint + "-p"] = np.abs(dct_vect)

	return dct_dict

# concatenates position, dct data to create a 1D feature vector
def create_feature_vect(exercise_data, dct_dict):

	feature_vect = []
	for joint in kinect_joints:

		position_data = exercise_data[joint + "-p"]
		for i in range(len(position_data)):
			feature_vect.append(position_data[i,0])
			feature_vect.append(position_data[i,1])
			feature_vect.append(position_data[i,2])

		dct_data = dct_dict[joint + "-p"]
		for i in range(len(dct_data)):
			feature_vect.append(dct_data[i])

	return np.array(feature_vect, dtype='float64')












