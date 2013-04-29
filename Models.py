import datetime
import numpy as np

from scipy.cluster.vq import kmeans, vq

class Model:

	"""
	Dependencies:

	-- Scipy 0.10.1
	"""

	def produce_initial_check_in_assignment(self, check_ins):
		"""
		Divides all check-ins into two clusters (Home and Work) 
		by their latitude and longitude.

		Returns two non-overlapping lists, one for Home and one for Work.

		check_ins -- list of check-ins, each of them being a dict with keys check_in_id,
		date, latitude, longitude, venue_id, check_in_message.
		"""
		for check_in in check_ins:
			if 'check_in_id' not in check_in:
				raise ValueError("Error: one of check-ins does not have ID!".format(fields=check_in))
			if 'latitude' not in check_in:
				raise ValueError("Error: check-in {id} does not have latitude!".format(id=check_in['check_in_id']))
			if 'longitude' not in check_in:
				raise ValueError("Error: check-in {id} does not have longitude!".format(id=check_in['check_in_id']))
		ids = [x["check_in_id"] for x in check_ins]
		if len(ids) != len(set(ids)):
			raise ValueError("Error: some check-ins have same IDs!")
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