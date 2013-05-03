import numpy as np
import random

from DataLoader import DataLoader
from Models import StanfordModel
from Utils import Utils


datasets = Utils.separate_dataset_by_days(DataLoader.load_check_ins_from_file(open("104665558.csv", 'U')))

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#days = ['Wednesday']

results = {}
for day in days:
	results[day] = []

for day in days:
	dataset = datasets["104665558"][day]

	random.shuffle(dataset)
	
	combinations = Utils.break_dataset_in_folds(dataset, 5)

	for combination in combinations:

		train = combination['train']
		test = combination['test'] 

		model = StanfordModel()
		model.train(train, number_of_iterations = 10)
		if model.parameters == None:
			continue

		correct = 0
		for check_in in test:
			real_venue = check_in["venue_id"]
			time = check_in["date"]
			predicted_venue = model.predict(time, train + test)
			if real_venue == predicted_venue:
				correct += 1
			#print real_venue + "\t" + predicted_venue

		results[day].append(float(correct)/len(test))

		#print "{day}\t{correct}/{total}\t{proportion}".format(day=day, correct=correct, total=len(test), proportion=float(correct)/len(test))

for day in days:
	results[day] = np.mean(results[day])

print results