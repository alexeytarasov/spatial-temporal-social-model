import math
import numpy as np
import random

from collections import Counter

from DataLoader import DataLoader
from Exceptions import TooSmallSingularValueError
from Models import StanfordModel, NCGModel, SocialModelStanford, CorrectSocialModelStanford, AdvancedSocialModel, SimpleSocialModel
from Utils import Utils

datasets = DataLoader.load_check_ins_from_directory("top_felix_users")

users = datasets.keys()

network = DataLoader.load_social_network(open("top_felix_users_connections.csv"))

"""friends = []
for user in datasets:
	if user == '104665558':
		continue
	friends.append(len(network[user]))
print np.min(friends)
print np.mean(friends)
print np.max(friends)
exit()"""

#print users
#exit()
#users = ["10221"]
#users = ['45474206', '276391406', '21913365', '27818171', '40557413', '19836108', '488667514', '94173972', '28668373', '33660680', '292750714', '104665558', '23209554', '549041707', '18488759', '82666753', '133067027', '30235429', '41234692', '29109326', '169585114', '14665537', '54670715', '258576072', '16332709', '83111133', '75911133', '573461782', '563315196', '111258523', '2365991', '24441491', '240102387'] 
#users = ['104665558', '23209554', '549041707', '18488759', '82666753', '133067027', '30235429', '41234692', '29109326', '169585114', '14665537', '54670715', '258576072', '16332709', '83111133', '75911133', '573461782', '563315196', '111258523', '2365991', '24441491', '240102387']

for user in users:

	#random.seed(1)

	global_results_radiation = []
	global_results_stanford = []
	global_results_stanford_social = []
	global_results_stanford_social_ours = []
	global_results_radiation_social = []
	global_results_radiation_social_ours = []
	global_results_only_social = []

	for i in range(0, 10):
			
		results_stanford = []
		results_stanford_social = []
		results_stanford_social_ours = []

		results_radiation = []
		results_radiation_social = []
		results_radiation_social_ours = []

		results_only_social = []

		dataset = datasets[user]

		by_days = Utils.separate_dataset_by_days({user: dataset})
		dataset = by_days[user]["Monday"] + by_days[user]["Tuesday"] + by_days[user]["Wednesday"] + by_days[user]["Thursday"]

		random.shuffle(dataset)
		
		combinations = Utils.break_dataset_in_folds(dataset, 5)

		for combination in combinations:

			social_model_stanford = CorrectSocialModelStanford()
			if user in network:
				social_model_ours = SimpleSocialModel(datasets, network[user], user)
			
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
				#------------
				if user in network:
					social_model_stanford.fit_model(model, network[user], datasets)
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						date = check_in["date"]
						latitude = check_in["latitude"]
						longitude = check_in["longitude"]

						spatio_temporal_p = model.predict(time, train + test, True)
						social_p = social_model_stanford.get_probabilities(network[user], datasets, date, train + test)

						total_p = {}
						for venue in spatio_temporal_p:
							total_p[venue] = spatio_temporal_p[venue] + social_p[venue]
						predicted_venue = max(total_p.iterkeys(), key=lambda k: total_p[k])
						if real_venue == predicted_venue:
							correct += 1
					results_stanford_social.append(float(correct)/len(test))
				else:
					results_stanford_social.append(-10.0)
				#-------
				if user in network:
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						date = check_in["date"]
						latitude = check_in["latitude"]
						longitude = check_in["longitude"]

						spatio_temporal_p = model.predict(time, train + test, True)
						social_p = social_model_ours.get_probabilities(network[user], datasets, date, train + test)

						total_p = {}
						for venue in spatio_temporal_p:
							total_p[venue] = spatio_temporal_p[venue] + social_p[venue]

						#print check_in["check_in_id"]
						#print spatio_temporal_p
						#print social_p
						#print total_p
						#print("stanford--------")

						predicted_venue = max(total_p.iterkeys(), key=lambda k: total_p[k])
						if real_venue == predicted_venue:
							correct += 1
					results_stanford_social_ours.append(float(correct)/len(test))
				else:
					results_stanford_social_ours.append(-10.0)
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

				if user in network:
					social_model_stanford.fit_model(model, network[user], datasets)
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						date = check_in["date"]
						latitude = check_in["latitude"]
						longitude = check_in["longitude"]

						spatio_temporal_p = model.predict(time, train + test, True)
						social_p = social_model_stanford.get_probabilities(network[user], datasets, date, train + test)

						total_p = {}
						for venue in spatio_temporal_p:
							total_p[venue] = spatio_temporal_p[venue] + social_p[venue]
						predicted_venue = max(total_p.iterkeys(), key=lambda k: total_p[k])
						if real_venue == predicted_venue:
							correct += 1
					results_radiation_social.append(float(correct)/len(test))
				else:
					results_radiation_social.append(-10.0)

				if user in network:
					#social_model_advanced.fit_model(model, network[user], datasets)
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						date = check_in["date"]
						latitude = check_in["latitude"]
						longitude = check_in["longitude"]

						spatio_temporal_p = model.predict(time, train + test, True)
						social_p = social_model_ours.get_probabilities(network[user], datasets, date, train + test)

						total_p = {}
						for venue in spatio_temporal_p:
							total_p[venue] = spatio_temporal_p[venue] + social_p[venue]

						#print check_in["check_in_id"]
						#print spatio_temporal_p
						#print social_p
						#print total_p
						#print("radiation--------")

						predicted_venue = max(total_p.iterkeys(), key=lambda k: total_p[k])
						if real_venue == predicted_venue:
							correct += 1
					results_radiation_social_ours.append(float(correct)/len(test))
				else:
					results_radiation_social_ours.append(-10.0)

				if user in network:
					#social_model_advanced.fit_model(model, network[user], datasets)
					correct = 0
					for check_in in test:
						real_venue = check_in["venue_id"]
						date = check_in["date"]
						latitude = check_in["latitude"]
						longitude = check_in["longitude"]

						total_p = social_model_ours.get_probabilities(network[user], datasets, date, train + test)

						#print check_in["check_in_id"]
						#print total_p
						#print("social--------")

						predicted_venue = max(total_p.iterkeys(), key=lambda k: total_p[k])
						if real_venue == predicted_venue:
							correct += 1
					results_only_social.append(float(correct)/len(test))
				else:
					results_only_social.append(-10.0)
			#---------------------------------------------------------------------------

		results_stanford = np.mean(results_stanford)
		results_stanford_social = np.mean(results_stanford_social)
		results_stanford_social_ours = np.mean(results_stanford_social_ours)
		results_radiation = np.mean(results_radiation)
		results_radiation_social = np.mean(results_radiation_social)
		results_radiation_social_ours = np.mean(results_radiation_social_ours)
		results_only_social = np.mean(results_only_social)

		#print "S" + "\t" + str(results_stanford)
		#print "R" + "\t" + str(results_radiation)

		global_results_stanford.append(results_stanford)
		global_results_radiation.append(results_radiation)
		global_results_stanford_social.append(results_stanford_social)
		global_results_stanford_social_ours.append(results_stanford_social_ours)
		global_results_radiation_social.append(results_radiation_social)
		global_results_radiation_social_ours.append(results_radiation_social_ours)
		global_results_only_social.append(results_only_social)

	print str(user) + "\t" + str(np.mean(global_results_stanford)) + "\t" + str(np.mean(global_results_stanford_social)) + "\t" + str(np.mean(global_results_stanford_social_ours)) + "\t" + str(np.mean(global_results_radiation)) + "\t" + str(np.mean(global_results_radiation_social)) + "\t" + str(np.mean(global_results_radiation_social_ours)) + "\t" + str(np.mean(results_only_social))