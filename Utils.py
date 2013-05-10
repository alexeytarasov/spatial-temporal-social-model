import matplotlib.pyplot as plt
import os

from sklearn import cross_validation

from DataLoader import *


class Utils:
	"""
	Dependencies:

	-- scikit-learn 0.11
	"""

	@staticmethod
	def break_dataset_in_folds(check_ins, folds):
		result = []
		if len(check_ins) < folds:
			raise ValueError("Error: the number of check-ins should be greater or equal then the number of folds!")
		k_fold = cross_validation.KFold(n=len(check_ins), k=folds, indices=True)
		for train_indices, test_indices in k_fold:
			test = [check_ins[x] for x in test_indices]
			train = [check_ins[x] for x in train_indices]
			result.append({"train": train, "test": test})
		return result


	@staticmethod
	def separate_dataset_by_days(check_ins):
		result = {}
		day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
		
		for user in check_ins:
			try:
				Utils.check_userless_check_in_list(check_ins[user])
			except ValueError as e:
				raise ValueError("Problem in dataset for user {user}: {original_exception}".format(user=user, original_exception=str(e)))
		
		for user in check_ins:
			result[user] = {}
			for day in day_list:
				result[user][day] = []

			for check_in in check_ins[user]:
				timestamp = check_in["date"]
				day_number = timestamp.date().weekday()
				result[user][day_list[day_number]].append(check_in)

		return result


	@staticmethod
	def check_check_in_syntax(check_in):
		"""
		Tests if check-in has all required attributes and they have valid values.

		Returns True if everything is OK, throws an exception otherwise.

		check_in -- a dict containing information about check-in. Should have at least
		venue_id, latitude, longitude, check_in_id and date keys. May contain any additional
		attributes as well. 
		"""
		if not isinstance(check_in, dict):
			raise ValueError("Check-in should be a dictionary!")
		check_in_attributes = check_in.keys()

		if 'check_in_id' not in check_in_attributes:
			raise ValueError("Check-in should contain check_in_id!")

		if 'venue_id' not in check_in_attributes:
			raise ValueError("Check-in should contain venue_id!")

		if 'latitude' not in check_in_attributes:
			raise ValueError("Error: check-in {id} does not have latitude!".format(id=check_in["check_in_id"]))
		try:
			float(check_in['latitude'])
		except (TypeError, ValueError):
			raise ValueError("Check-in has invalid latitude!")

		if 'longitude' not in check_in_attributes:
			raise ValueError("Error: check-in {id} does not have longitude!".format(id=check_in["check_in_id"]))
		try:
			float(check_in['longitude'])
		except (TypeError, ValueError):
			raise ValueError("Check-in has invalid longitude!")

		if 'date' not in check_in_attributes:
			raise ValueError("Check-in should contain date!")

		return True

	@staticmethod
	def check_userless_check_in_list(check_ins):
		if not isinstance(check_ins, list):
			raise ValueError("Error: the input argument is not a valid list!")
		if len(check_ins) == 0:
			raise ValueError("Error: the list of check-ins is empty!")
		#if len(check_ins) == 1:
		#	raise ValueError("Error: the list should contain at least two check-ins!")
		for check_in in check_ins:
			Utils.check_check_in_syntax(check_in)
		ids = [x["check_in_id"] for x in check_ins]
		if len(ids) != len(set(ids)):
			raise ValueError("Error: some check-ins have same IDs!")

	@staticmethod
	def plot_check_ins(check_ins, directory):
		if not os.path.isdir(directory):
			os.mkdir(directory)
		for user in check_ins:
			latitudes = []
			longitudes = []
			for check_in in check_ins[user]:
				latitudes.append(check_in["latitude"])
				longitudes.append(check_in["longitude"])
			fig = plt.figure()
			fig.suptitle(user + ", " + str(len(latitudes)))
			plt.scatter(latitudes, longitudes)
			plt.savefig(directory + "/" + user + ".png")
			

if __name__ == "__main__":
	datasets = DataLoader.load_check_ins_from_directory("top_felix_users")
	Utils.plot_check_ins(datasets, "check_in_graphs")