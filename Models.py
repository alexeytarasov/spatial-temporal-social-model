import datetime
import numpy as np
import time

from scipy.cluster.vq import kmeans, vq
from Utils import Utils

class Model(object):

	"""
	An abstract class for a model to predict user's location.

	Dependencies:

	-- NumPy 1.6.1
	-- SciPy 0.10.1
	"""

	def produce_initial_check_in_assignment(self, check_ins):
		"""
		Divides all check-ins into two clusters (Home and Work) 
		by their latitude and longitude.

		Returns two non-overlapping lists, one for Home and one for Work.

		check_ins -- list of check-ins, each of them being a dict with keys check_in_id,
		date, latitude, longitude, venue_id, check_in_message.
		"""
		Utils.check_userless_check_in_list(check_ins)
		datapoints = np.array([[x['latitude'], x['longitude']] for x in check_ins])
		ids = [x['check_in_id'] for x in check_ins]
		centroids,_ = kmeans(datapoints, 2)
		labels,_ = vq(datapoints,centroids)
		home_check_ins = []
		work_check_ins = []
		for i in range(0, len(datapoints)):
			check_in = [x for x in check_ins if x["check_in_id"] == ids[i]][0]
			if labels[i] == 0:
				home_check_ins.append(check_in)
			else:
				work_check_ins.append(check_in)
		return home_check_ins, work_check_ins


	def produce_initial_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		"""
		Produces initial estimates of model parameters via maximum likelihood estimation.

		Returns a dict where keys are parameter names and values are parameter values.

		check_ins_H -- list of home check-ins
		check_ins_W -- list of work check-ins
		"""
		if not isinstance(check_ins_H, list):
			raise ValueError("First argument has to be a list!")
		if not isinstance(check_ins_W, list):
			raise ValueError("Second argument has to be a list!")
		if len(check_ins_H) == 0:
			raise ValueError("First list has to contain at least one check-in!")
		if len(check_ins_W) == 0:
			raise ValueError("Second list has to contain at least one check-in!")
		Utils.check_userless_check_in_list(check_ins_H)
		Utils.check_userless_check_in_list(check_ins_W)
	

class StanfordModel(Model):
	"""
	Model from "Friendship and Mobility: User Movement In Location-Based Social Networks" 
	by E. Cho, S. A. Myers, J. Leskovec. Procs of KDD, 2011.
	"""

	def produce_initial_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		super(StanfordModel, self).produce_initial_max_likelihood_estimates(None, None)


#model = StanfordModel()
#model.produce_initial_max_likelihood_estimates(None, None)