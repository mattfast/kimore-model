import os
import xlrd
import csv
import json

kinect_joints = ["spinebase", "spinemid", "neck", "head", "shoulderleft", "elbowleft", "wristleft", "handleft", "shoulderright", "elbowright", "wristright", "handright", "hipleft", "kneeleft", "ankleleft", "footleft", "hipright", "kneeright", "ankleright", "footright", "spineshoulder", "handtipleft", "thumbleft", "handtipright", "thumbright"]

def load_kimore_data(path):

	data = []
	for (root, dirs, files) in os.walk(path):

		# if current directory contains "Raw", extract data
		if "Raw" in dirs:

			new_dict = {}

			# get exercise number
			new_dict["Exercise"] = int(root[-1])

			print("Working on " + root)

			# extract raw data
			raw_files = os.listdir(os.path.join(root, "Raw"))
			for file in raw_files:

				file_path = os.path.join(os.path.join(root, "Raw"),file)
				csv_file = open(file_path, newline='')
				csv_reader = csv.reader(csv_file)

				if file.startswith("JointOrientation"):

					for joint in kinect_joints:
						new_dict[joint + "-o"] = []

					for row in csv_reader:
						for i in range(len(kinect_joints)):
							if len(row) > 0:
								new_dict[kinect_joints[i] + "-o"].append(row[(4*i):(4*i+4)])

					orientation_present = True

				elif file.startswith("JointPosition"):

					for joint in kinect_joints:
						new_dict[joint + "-p"] = []

					for row in csv_reader:
						for i in range(len(kinect_joints)):
							if len(row) > 0:
								new_dict[kinect_joints[i] + "-p"].append(row[(4*i):(4*i+3)])

				elif file.startswith("TimeStamp"):

					new_dict["Timestamps"] = []
					for row in csv_reader:
						if len(row) > 0:
							new_dict["Timestamps"].append(row[0])

			# verify that all data was collected
			if 'spinebase-o' not in new_dict:
				continue

			if 'spinebase-p' not in new_dict:
				continue

			if 'Timestamps' not in new_dict:
				continue


			# extract data labels
			label_files = os.listdir(os.path.join(root, "Label"))
			for file in label_files:

				file_path = os.path.join(os.path.join(root, "Label"),file)
				book = xlrd.open_workbook(file_path)
				sheet = book.sheet_by_index(0)

				titles = sheet.row_values(0)
				vals = sheet.row_values(1)

				if file.startswith("SuppInfo"):
					for t, v in zip(titles, vals):
						new_dict[t] = v

				elif file.startswith("ClinicalAssessment"):
					new_dict["cTS"] = vals[new_dict["Exercise"]]
					new_dict["cPO"] = vals[new_dict["Exercise"] + 5]
					new_dict["cCF"] = vals[new_dict["Exercise"] + 10]

			# append exercise to data
			data.append(new_dict)

	return data

def add_binary_labels(kimore_data, path):

	book = xlrd.open_workbook(path)
	sheet = book.sheet_by_index(0)

	for i in range(1,sheet.nrows):
		vals = sheet.row_values(i)

		name = vals[0]
		labels = []
		for j in range(1,len(vals)):
			if vals[j] == 'Y':
				labels.append(1)
			elif vals[j] == 'N' or vals[j] == 'U':
				labels.append(0)
			else:
				raise ValueError("Invalid label")

		kimore_dict = next((ex for ex in kimore_data if ex["Subject ID"] == name and ex["Exercise"] == 5), None)
		if kimore_dict is None:
			print(name + " NOT PRESENT")
		else:
			kimore_dict["Critiques"] = labels
			print(name)
			print(kimore_dict["Critiques"])

	return kimore_data


def json_encode(data, path):

	f = open(path, "w+")
	f.write(json.dumps(data))
	f.close()

def json_decode(path):
	
	if (os.path.exists(path)):
		f = open(path, "r")
		contents = f.read()
		data = json.loads(contents)
	else:
		raise ValueError("Specified file does not exist")

	return data



