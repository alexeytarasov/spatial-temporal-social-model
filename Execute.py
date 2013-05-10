import math
import numpy as np
import random

from collections import Counter

from DataLoader import DataLoader
from Exceptions import TooSmallSingularValueError
from Models import StanfordModel, NCGModel
from Utils import Utils


global_results_stanford = []
global_results_radiation = []

#datasets = DataLoader.load_check_ins_from_directory("top_brightkite_users")
datasets = DataLoader.load_check_ins_from_directory("top_felix_users")
users = datasets.keys()

#print users
#exit()
users = ["24441491"]
#users = ['45474206', '276391406', '21913365', '27818171', '40557413', '19836108', '488667514', '94173972', '28668373', '33660680', '292750714', '104665558', '23209554', '549041707', '18488759', '82666753', '133067027', '30235429', '41234692', '29109326', '169585114', '14665537', '54670715', '258576072', '16332709', '83111133', '75911133', '573461782', '563315196', '111258523', '2365991', '24441491', '240102387'] 

for user in users:

	#random.seed(1)

	global_results_radiation = []
	global_results_stanford = []

	for i in range(0, 10):

		results_stanford = {}
		results_radiation = {}
			
		results_stanford = []
		results_radiation = []

		dataset = datasets[user]

		by_days = Utils.separate_dataset_by_days({user: dataset})
		dataset = by_days[user]["Monday"] + by_days[user]["Tuesday"] + by_days[user]["Wednesday"] + by_days[user]["Thursday"]

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
				results_stanford.append(float(correct)/len(test))
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

				results_radiation.append(float(correct)/len(test))
			#---------------------------------------------------------------------------

		results_stanford = np.mean(results_stanford)
		results_radiation = np.mean(results_radiation)

		#print "S" + "\t" + str(results_stanford)
		#print "R" + "\t" + str(results_radiation)

		global_results_stanford.append(results_stanford)
		global_results_radiation.append(results_radiation)


	print str(user) + "\t" + str(np.mean(global_results_stanford)) + "\t" + str(np.mean(global_results_radiation))