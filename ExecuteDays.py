import math
import numpy as np
import random

from collections import Counter

from DataLoader import DataLoader
from Models import StanfordModel, NCGModel
from Utils import Utils


global_results_stanford = []
global_results_radiation = []

datasets = Utils.separate_dataset_by_days(DataLoader.load_check_ins_from_file(open("104665558.csv", 'U')))
users = datasets.keys()
users = ["104665558"]

for user in users:

	for i in range(0, 1):

		days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
		#days = ['Wednesday']

		results_stanford = {}
		results_radiation = {}
		for day in days:
			results_stanford[day] = []
			results_radiation[day] = []

		for day in days:
			dataset = datasets[user][day]

			random.shuffle(dataset)
			
			combinations = Utils.break_dataset_in_folds(dataset, 5)

			for combination in combinations:

				train = combination['train']
				test = combination['test'] 

				#---------------------------------------------------------------------------
				model = StanfordModel()
				model.train(train, number_of_iterations = 10)
				if model.parameters != None:
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						time = check_in["date"]
						predicted_venue = model.predict(time, train + test)
						if real_venue == predicted_venue:
							correct += 1
					results_stanford[day].append(float(correct)/len(test))
				#---------------------------------------------------------------------------
				all_check_ins = train + test
				all_venues = [x["venue_id"] for x in all_check_ins]
				n_values = Counter(all_venues)
				coordinates = {}
				for venue in all_venues:
					latitude = np.median([x["latitude"] for x in all_check_ins if x["venue_id"] == venue])
					longitude = np.median([x["longitude"] for x in all_check_ins if x["venue_id"] == venue])
					coordinates[venue] = (latitude, longitude)

				model = NCGModel(n_values, coordinates)
				model.train(train, number_of_iterations = 10)
				if model.parameters != None:
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						time = check_in["date"]
						predicted_venue = model.predict(time, train + test)
						if real_venue == predicted_venue:
							correct += 1

					results_radiation[day].append(float(correct)/len(test))
				#---------------------------------------------------------------------------

		for day in days:
			results_stanford[day] = np.mean(results_stanford[day])
			results_radiation[day] = np.mean(results_radiation[day])


		global_results_stanford.append(results_stanford)
		global_results_radiation.append(results_radiation)
		print results_stanford
		print results_radiation


	for day in global_results_stanford[0]:
		day_results_stanford = []
		day_results_radiation = []
		for x in range(0, len(global_results_stanford)):
			if math.isnan(global_results_stanford[x][day]):
				day_results_stanford.append(0)
			else:
				day_results_stanford.append(global_results_stanford[x][day])

			if math.isnan(global_results_radiation[x][day]):
				day_results_radiation.append(0)
			else:
				day_results_radiation.append(global_results_radiation[x][day])
		print str(user) + "\t" + day + "\t" + str(np.mean(day_results_stanford)) + "\t" + str(np.mean(day_results_radiation))