import datetime
import math
import numpy as np
import random
import time

from scipy.cluster.vq import kmeans, vq
from scipy.stats import morestats
from Exceptions import TooSmallSingularValueError
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


	def produce_max_likelihood_estimates(self, check_ins_H, check_ins_W):
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


	@staticmethod
	def hours_from_midnight(timestamp):
		"""
		Calculates the number of hours since midnight for timestamp. For 14:38:00 
		should return ~14.716.

		Returns a float, representing a number of hours since midnight.

		timestamp -- datetime.datetime object, for which the value is calculated.
		"""
		if not isinstance(timestamp, datetime.datetime):
			raise ValueError("Timestamp should be an instance of datetime.datetime!")
		result = float(timestamp.hour) + (float(timestamp.minute) / 60)
		return result


	@staticmethod
	def calculate_circular_mean(values_in_hours):
		"""
		Calculates circular mean for timestamps, expressed as number of hours since
		midnight.

		Returns a float, representing a circular mean.

		values_in_hours -- list of timestamps, expressed as number of hours since 
		midnight.
		"""
		if not isinstance(values_in_hours, list):
			raise ValueError("Timestamps should be a list!")
		for i in values_in_hours:
			if not isinstance(i, (int, long, float)):
				raise ValueError("{value} is not a number!".format(value=i))
		values_in_radians = []
		for value in values_in_hours:
			values_in_radians.append((float(value) / 24) * (math.pi * 2))
		return morestats.circmean(np.array(values_in_radians)) / (math.pi * 2) * 24


	@staticmethod
	def calculate_covariation_matrix(check_ins):
		"""
		Calculates covariation matrix for latitude and longitude.

		Returns a covariation matrix.

		check_ins -- list of check-ins, for which covariation is calculated.
		"""
		if not isinstance(check_ins, list):
			raise ValueError("Error: a list should be supplied as an algorithm!")
		for check_in in check_ins:
			Utils.check_check_in_syntax(check_in)
		Utils.check_userless_check_in_list(check_ins)
		pairs = []
		for check_in in check_ins:
			pair = []
			pair.append(check_in["latitude"])
			pair.append(check_in["longitude"])
			pairs.append(pair)
		data_matrix = np.array(pairs).T
		result = np.cov(data_matrix)
		return result


	@staticmethod
	def calculate_circular_SD(values_in_hours):
		"""
		Calculates circular standard deviation for timestamps, expressed as number 
		of hours since midnight.

		Returns a float, representing a circular standard deviation.

		values_in_hours -- list of timestamps, expressed as number of hours since 
		midnight.
		"""
		if not isinstance(values_in_hours, list):
			raise ValueError("Timestamps should be a list!")
		for i in values_in_hours:
			if not isinstance(i, (int, long, float)):
				raise ValueError("{value} is not a number!".format(value=i))
		values_in_radians = []
		for value in values_in_hours:
			values_in_radians.append((float(value) / 24) * (math.pi * 2))

		return (morestats.circstd(np.array(values_in_radians)) / (math.pi * 2)) * 24


	def calculate_temporal_probabilities(self, t):
		"""
		Calculates temporal probabilities of belonging to Home and Work clusters.

		Returns a tuple (Home, Work) with probabilities.

		t -- time of check-in, for which probabilities are calculated.
		"""
		N_h = self.calculate_N(t, self.parameters["Pc_h"], self.parameters["sigma_h"], self.parameters["tau_h"])
		N_w = self.calculate_N(t, self.parameters["Pc_w"], self.parameters["sigma_w"], self.parameters["tau_w"])
		P_h = N_h / float(N_h + N_w)
		P_w = N_w / float(N_h + N_w)
		return P_h, P_w


	def calculate_spatial_probabilities(self, t):
		"""
		Calculates spatial probabilities of belonging to Home and Work clusters.

		Returns a tuple (Home, Work) with probabilities.

		t -- time of check-in, for which probabilities are calculated.
		"""
		None


	@staticmethod
	def calculate_N(t_hours, Pc, sigma, tau):
		"""
		Calculates a probability that the time belongs to the cluster represented by parameters.

		Returns a float, representing the probability.

		t_hours -- time for which probability has to be calculated, measured in hours since 
		midnight.
		Pc, sigma, tau -- values from "Friendship and Mobility: User Movement In Location-Based Social Networks" 
		by E. Cho, S. A. Myers, J. Leskovec. Procs of KDD, 2011.
		"""
		if t_hours > 24 or t_hours < 0:
			raise ValueError("Error: t_hours should be in the [0, 24] interval!")
		if Pc < 0 or Pc > 1:
			raise ValueError("Error: Pc should be in the [0, 1] interval!")
		if sigma < 0 or sigma > 24:
			raise ValueError("Error: sigma should be in the [0, 24] interval!")
		if tau < 0 or tau > 24:
			raise ValueError("Error: tau should be in the [0, 24] interval!")
		if sigma == 0:
			if t_hours == tau:
				return 1.0
			else:
				return 0.0
		t = t_hours
		first_multiplier = float(Pc) / math.sqrt(2 * math.pi * (sigma ** 2))
		power = (-(float(math.pi) / 12) ** 2) * (((t - tau) ** 2) / (2 * (sigma ** 2)))
		return first_multiplier * math.exp(power)


	def train(self, number_of_iterations):
		None


	def perform_next_iteration(self):
		None


	def predict(self, t):
		None

	
class StanfordModel(Model):
	"""
	Model from "Friendship and Mobility: User Movement In Location-Based Social Networks" 
	by E. Cho, S. A. Myers, J. Leskovec. Procs of KDD, 2011.
	"""

	@staticmethod
	def probability_multivariate_normal(x, mean, covariance_matrix):
		"""
		Calculates the probability for multivariate Gaussian.

		Returns a probability (unnormalised, can be more than 1).

		x -- input vector.
		mean -- vector of means for the distribution.
		covariance_matrix -- covariance matrix for the distribution.
		"""
		cov = covariance_matrix
		if np.min(np.linalg.svd(cov)[1]) < 10**(-7) or np.linalg.det(cov) == 0:
			raise TooSmallSingularValueError()
		det = np.linalg.det(cov)
		first_multiplier = float(1) / ((2 * math.pi) ** 2 * (det ** 0.5))
		power = -0.5 * np.dot(np.dot(np.subtract(np.array(x), np.array(mean)), np.linalg.inv(cov)), np.subtract(np.array(x), np.array(mean)))
		try:
			return first_multiplier * math.exp(power)
		except FloatingPointError:
			return 0


	def produce_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		super(StanfordModel, self).produce_max_likelihood_estimates(check_ins_H, check_ins_W)
		result = {}
		home_times = [self.hours_from_midnight(x['date']) for x in check_ins_H]
		work_times = [self.hours_from_midnight(x['date']) for x in check_ins_W]
		# Temporal parameters
		result["tau_h"] = self.calculate_circular_mean(home_times)
		result["tau_w"] = self.calculate_circular_mean(work_times)
		result["sigma_h"] = self.calculate_circular_SD(home_times)
		result["sigma_w"] = self.calculate_circular_SD(work_times)
		# Spatial parameters
		home_latitudes = [x['latitude'] for x in check_ins_H]
		work_latitudes = [x['latitude'] for x in check_ins_W]
		home_longitudes = [x['longitude'] for x in check_ins_H]
		work_longitudes = [x['longitude'] for x in check_ins_W]
		result['mju_h'] = [np.mean(home_latitudes), np.mean(home_longitudes)]
		result['mju_w'] = [np.mean(work_latitudes), np.mean(work_longitudes)]
		result['Sigma_h'] = self.calculate_covariation_matrix(check_ins_H)
		result['Sigma_w'] = self.calculate_covariation_matrix(check_ins_W)
		return result


	def calculate_spatial_probabilities(self, latitude, longitude):
		"""
		Calculates spatial probabilities of belonging to Home and Work clusters using Gaussians.

		Returns a tuple (Home, Work) with probabilities.

		t -- time of check-in, for which probabilities are calculated.
		"""
		P_h_not_normalised = self.probability_multivariate_normal([latitude, longitude], self.parameters["mju_h"], self.parameters["Sigma_h"])
		P_w_not_normalised = self.probability_multivariate_normal([latitude, longitude], self.parameters["mju_w"], self.parameters["Sigma_w"])
		return P_h_not_normalised, P_w_not_normalised


	def calculate_likelihood(self):
		likelihood = 0
		for check_in in self.P_total_H:
			P_h = self.P_total_H[check_in]
			P_w = self.P_total_W[check_in]
			likelihood += math.log(P_h + P_w)
		return likelihood


	def train(self, check_ins, number_of_iterations = 10, number_of_models = 10):
		models = {}
		for j in range(0, number_of_models):
			self.check_ins_h, self.check_ins_w = self.produce_initial_check_in_assignment(check_ins)
			self.check_ins = self.check_ins_h + self.check_ins_w
			for i in range(0, number_of_iterations):
				#print "Iteration {i}".format(i=i)
				before_h = [x["check_in_id"] for x in self.check_ins_h]
				self.perform_next_iteration()
				if self.parameters == None:
					break
				after_h = [x["check_in_id"] for x in self.check_ins_h]
				if set(before_h) == set(after_h):
					break
			if self.parameters != None:
				models[j] = {}
				models[j]["parameters"] = self.parameters
				models[j]["likelihood"] = self.calculate_likelihood()
		
		if len(models) > 0:
			max_likelihood = np.max([models[x]["likelihood"] for x in models])
			max_likelihood_models = [models[x]["parameters"] for x in models if models[x]["likelihood"] == max_likelihood]
			random.shuffle(max_likelihood_models)
			self.parameters = max_likelihood_models[0]


	def aggregated_probability(self, check_in):
		"""
		Calculates the probability for the check-in to belong to home or work cluster
		considering all components of the model.

		Returns a tuple of probabilities of belonging to cluster H or W.

		check_in -- input check-in.
		"""
		Utils.check_check_in_syntax(check_in)
		id = str(check_in["check_in_id"])
		P_home_cumulative = self.P_temporal_H[id] * self.P_spatial_H[id]
		P_work_cumulative = self.P_temporal_W[id] * self.P_spatial_W[id]
		return P_home_cumulative, P_work_cumulative


	def perform_next_iteration(self):
		self.parameters = self.produce_max_likelihood_estimates(self.check_ins_h, self.check_ins_w)
		self.parameters["Pc_h"] = len(self.check_ins_h) / float(len(self.check_ins_h) + len(self.check_ins_w))
		self.parameters["Pc_w"] = len(self.check_ins_h) / float(len(self.check_ins_h) + len(self.check_ins_w))
		self.P_temporal_H = {}
		self.P_temporal_W = {}
		self.P_spatial_H = {}
		self.P_spatial_W = {}
		for check_in in self.check_ins:
			id = str(check_in["check_in_id"])
			t = self.hours_from_midnight(check_in["date"])
			self.P_temporal_H[id], self.P_temporal_W[id] = self.calculate_temporal_probabilities(t)
			
			latitude = check_in["latitude"]
			longitude = check_in["longitude"]
			try:
				self.P_spatial_H[id], self.P_spatial_W[id] = self.calculate_spatial_probabilities(latitude, longitude)
			except TooSmallSingularValueError:
				self.parameters = None
				return

		self.check_ins_h = []
		self.check_ins_w = []

		self.P_total_H = {}
		self.P_total_W = {}

		for check_in in self.check_ins:
			home, work = self.aggregated_probability(check_in)
			self.P_total_H[check_in["check_in_id"]] = home
			self.P_total_W[check_in["check_in_id"]] = work
			if home > work:
				self.check_ins_h.append(check_in)
			else:
				self.check_ins_w.append(check_in)

		if self.check_ins_h == [] or self.check_ins_w == []:
			self.parameters = None
			return


	def _get_average_venue_coordinates(self, check_ins):
		result = {}
		all_venues = set([x["venue_id"] for x in check_ins])
		for venue in all_venues:
			latitudes = [x["latitude"] for x in check_ins if x["venue_id"] == venue]
			longitudes = [x["longitude"] for x in check_ins if x["venue_id"] == venue]
			result[venue] = {}
			result[venue]["latitude"] = np.median(latitudes)
			result[venue]["longitude"] = np.median(longitudes)
		return result


	def predict(self, timestamp, check_ins):
		t = self.hours_from_midnight(timestamp)
		venue_coordinates = self._get_average_venue_coordinates(check_ins)
		venue_probabilities = {}

		for venue in venue_coordinates:
			latitude = venue_coordinates[venue]["latitude"]
			longitude = venue_coordinates[venue]["longitude"]
			P_temporal_h, P_temporal_w = self.calculate_temporal_probabilities(t)
			P_spatial_h, P_spatial_w = self.calculate_spatial_probabilities(latitude, longitude)
			if P_temporal_h > P_temporal_w:
				venue_probabilities[venue] = P_spatial_h
			else:
				venue_probabilities[venue] = P_spatial_w

		return max(venue_probabilities.iterkeys(), key=lambda k: venue_probabilities[k])