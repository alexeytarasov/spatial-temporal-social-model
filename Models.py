import datetime
import math
import networkx as nx
import numpy as np
import operator
import random
import time

from copy import deepcopy
from rpy2 import robjects
from rpy2.robjects.packages import importr
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
	-- rpy2 2.2.1 + R 2.15.3 + mnormt 1.4-5
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
		
		home_check_ins = []
		work_check_ins = []

		all_check_ins = deepcopy(check_ins)
		random.shuffle(all_check_ins)
		home_check_ins = all_check_ins[:len(check_ins) / 2]
		work_check_ins = all_check_ins[len(check_ins) / 2:]

		return home_check_ins, work_check_ins

		"""Utils.check_userless_check_in_list(check_ins)
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
		return home_check_ins, work_check_ins"""


	def assign_initial_check_in_assignment(self, check_ins):
		home, work = self.produce_initial_check_in_assignment(check_ins)
		self.check_ins_h = home
		self.check_ins_w = work


	def produce_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		"""
		Produces initial estimates of model parameters via maximum likelihood estimation.

		Returns a dict where keys are parameter names and values are parameter values.

		check_ins_H -- list of home check-ins
		check_ins_W -- list of work check-ins
		"""
		None


	@staticmethod
	def check_max_likelihood_estimates_input(check_ins_H, check_ins_W):
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
		try:
			result = np.cov(data_matrix)
		except Exception as e:
			print str(e)
			print data_matrix
			exit()
		singular_values = np.linalg.svd(np.array(result))[1].tolist()
		min_singular_value = singular_values[len(singular_values) - 1]
		if min_singular_value < 10 ** (-7):
			#print data_matrix
			#print singular_values
			raise TooSmallSingularValueError()
		#print data_matrix
		#print singular_values
		#print result.tolist()
		#print("--------")
		return result.tolist()



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

	def __init__(self):
		self.parameters = {}


	@staticmethod
	def probability_multivariate_normal(x, mean, covariance_matrix):
		"""
		Calculates the probability for multivariate Gaussian.

		Returns a probability (unnormalised, can be more than 1).

		x -- input vector.
		mean -- vector of means for the distribution.
		covariance_matrix -- covariance matrix for the distribution.
		"""

		# Check if covariation matrix singular values are OK
		singular_values = np.linalg.svd(np.array(covariance_matrix))[1].tolist()
		min_singular_value = singular_values[len(singular_values) - 1]
		#if min_singular_value < 10 ** (-7) or math.isnan(covariance_matrix[0][0]):
		#	raise TooSmallSingularValueError()

		robjects.r('library(mnormt)')
		x_r = robjects.r('x <- cbind(' + str(x[0]) + ", " + str(x[1]) +')')
		covariance_matrix_string = "sigma <- matrix(c("
		covariance_matrix_string += str(covariance_matrix[0][0]) + ", "
		covariance_matrix_string += str(covariance_matrix[0][1]) + ", "
		covariance_matrix_string += str(covariance_matrix[1][0]) + ", "
		covariance_matrix_string += str(covariance_matrix[1][1]) + "), 2, 2)"
		covariance_matrix_r = robjects.r(covariance_matrix_string)
		mean_r = robjects.r('mu <- cbind(' + str(mean[0]) +", " + str(mean[1]) + ')')
		pmnorm = robjects.r("pmnorm")
		result = pmnorm(x_r, mean_r, covariance_matrix_r)[0]
		return result


	def produce_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		#if len(check_ins_H) < 2 or len(check_ins_W) < 2:
		#	raise OneCheckInInCluster()
		#print len(check_ins_H)
		#print len(check_ins_W)
		self.check_max_likelihood_estimates_input(check_ins_H, check_ins_W)
		result = {}
		home_times = [self.hours_from_midnight(x['date']) for x in check_ins_H]
		work_times = [self.hours_from_midnight(x['date']) for x in check_ins_W]
		# Temporal parameters
		result["tau_h"] = self.calculate_circular_mean(home_times)
		result["tau_w"] = self.calculate_circular_mean(work_times)
		result["sigma_h"] = self.calculate_circular_SD(home_times)
		if result["sigma_h"] < 10 ** (-4):
			result["sigma_h"] = 10 ** (-4)
		result["sigma_w"] = self.calculate_circular_SD(work_times)
		if result["sigma_w"] < 10 ** (-4):
			result["sigma_w"] = 10 ** (-4)
		# Spatial parameters
		home_latitudes = [x['latitude'] for x in check_ins_H]
		work_latitudes = [x['latitude'] for x in check_ins_W]
		home_longitudes = [x['longitude'] for x in check_ins_H]
		work_longitudes = [x['longitude'] for x in check_ins_W]
		result['mju_h'] = [np.mean(home_latitudes), np.mean(home_longitudes)]
		result['mju_w'] = [np.mean(work_latitudes), np.mean(work_longitudes)]
		result['Sigma_h'] = self.calculate_covariation_matrix(check_ins_H)
		result['Sigma_w'] = self.calculate_covariation_matrix(check_ins_W)
		#print result['Sigma_h']
		#print result['Sigma_w']
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
			self.assign_initial_check_in_assignment(check_ins)
			self.check_ins = self.check_ins_h + self.check_ins_w
			for i in range(0, number_of_iterations):
				#print "Iteration {i}".format(i=i)
				before_h = [x["check_in_id"] for x in self.check_ins_h]
				try:	
					self.perform_next_iteration()
				except TooSmallSingularValueError():
					self.parameters == None
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
			#print models
			likelihoods = [models[x]["likelihood"] for x in models if not math.isnan(models[x]["likelihood"])]
			if len(likelihoods) > 0:
				max_likelihood = np.max(likelihoods)
				max_likelihood_models = [models[x]["parameters"] for x in models if models[x]["likelihood"] == max_likelihood]
				random.shuffle(max_likelihood_models)
				#print("--------")
				self.parameters = max_likelihood_models[0]
			else:
				self.parameters = None


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

		#print "Start of iteration:"
		#print [x["check_in_id"] for x in self.check_ins_h]
		#print [x["check_in_id"] for x in self.check_ins_w]

		if len(self.check_ins_h) == 1 or len(self.check_ins_w) == 1:
			self.parameters = None
			return

		if self.parameters == None:
			self.parameters = {}
		try:
			self.parameters.update(self.produce_max_likelihood_estimates(self.check_ins_h, self.check_ins_w))
		except TooSmallSingularValueError:
			self.parameters = None
			return
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

		#print "-" + str(len(self.check_ins_h))
		#print "-" + str(len(self.check_ins_w))

		#print "End of iteration:"
		#print [x["check_in_id"] for x in self.check_ins_h]
		#print [x["check_in_id"] for x in self.check_ins_w]

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


	def predict(self, timestamp, check_ins, return_all_probabilities = False):
		t = self.hours_from_midnight(timestamp)
		venue_coordinates = self._get_average_venue_coordinates(check_ins)
		venue_probabilities = {}

		for venue in venue_coordinates:
			latitude = venue_coordinates[venue]["latitude"]
			longitude = venue_coordinates[venue]["longitude"]
			P_temporal_h, P_temporal_w = self.calculate_temporal_probabilities(t)
			P_spatial_h, P_spatial_w = self.calculate_spatial_probabilities(latitude, longitude)
			#if P_temporal_h > P_temporal_w:
			#	venue_probabilities[venue] = P_spatial_h
			#else:
			#	venue_probabilities[venue] = P_spatial_w
			venue_probabilities[venue] = (P_temporal_h * P_spatial_h) + (P_temporal_w * P_spatial_w)

		if return_all_probabilities == True:
			return venue_probabilities

		return max(venue_probabilities.iterkeys(), key=lambda k: venue_probabilities[k])


class NCGModel(StanfordModel):


	def __init__(self, n_values, coordinates):
		self.n_values = n_values
		self.coordinates = coordinates


	def predict(self, timestamp, check_ins, return_all_probabilities = False):
		t = self.hours_from_midnight(timestamp)
		venue_coordinates = self._get_average_venue_coordinates(check_ins)
		venue_probabilities = {}

		all_venues = [x["venue_id"] for x in check_ins]

		for venue in venue_coordinates:
			latitude = venue_coordinates[venue]["latitude"]
			longitude = venue_coordinates[venue]["longitude"]
			P_temporal_h, P_temporal_w = self.calculate_temporal_probabilities(t)
			
			P_spatial_h = self.p_radiation(self.parameters["central_h"], self.parameters["m_h"], venue, all_venues)
			P_spatial_w = self.p_radiation(self.parameters["central_w"], self.parameters["m_w"], venue, all_venues)

			venue_probabilities[venue] = (P_temporal_h * P_spatial_h) + (P_temporal_w * P_spatial_w)

		if return_all_probabilities == True:
			return venue_probabilities
		else:
			return max(venue_probabilities.iterkeys(), key=lambda k: venue_probabilities[k])


	def perform_next_iteration(self):
		if self.parameters == None:
			self.parameters = {}
		self.parameters.update(self.produce_max_likelihood_estimates(self.check_ins_h, self.check_ins_w))
		self.parameters["Pc_h"] = len(self.check_ins_h) / float(len(self.check_ins_h) + len(self.check_ins_w))
		self.parameters["Pc_w"] = len(self.check_ins_h) / float(len(self.check_ins_h) + len(self.check_ins_w))
		
		venues_h = [x["venue_id"] for x in self.check_ins_h]
		self.parameters["central_h"] = self.get_central_venue(venues_h, self.parameters["m_h"])
		venues_w = [x["venue_id"] for x in self.check_ins_w]
		self.parameters["central_w"] = self.get_central_venue(venues_w, self.parameters["m_w"])
		all_venues = venues_h + venues_w

		self.parameters["m_h"] = self.get_m(self.parameters["m_h"], self.parameters["central_h"], venues_h)
		self.parameters["m_w"] = self.get_m(self.parameters["m_w"], self.parameters["central_w"], venues_w)

		# Spatial probabilities
		self.P_spatial_H = {}
		self.P_spatial_W = {}
		for venue in set(all_venues):
			self.P_spatial_H[venue] = self.p_radiation(self.parameters["central_h"], self.parameters["m_h"], venue, all_venues)
			self.P_spatial_W[venue] = self.p_radiation(self.parameters["central_w"], self.parameters["m_w"], venue, all_venues)
		sum_H = np.sum(self.P_spatial_H.values())
		sum_W = np.sum(self.P_spatial_W.values())
		#print self.P_spatial_H.values()
		#print self.P_spatial_W.values()
		for venue in set(all_venues):
			if float(sum_H) == 0:
				self.P_spatial_H[venue] = 0
			else:
				self.P_spatial_H[venue] /= float(sum_H)
			if float(sum_W) == 0:
				self.P_spatial_W[venue] = 0
			else:
				self.P_spatial_W[venue] /= float(sum_W)

		# Temporal probabilities
		self.P_temporal_H = {}
		self.P_temporal_W = {}

		for venue in set(all_venues):
			P_temporal_H = 0
			P_temporal_W = 0
			check_ins_in_venue = [x for x in self.check_ins if x["venue_id"] == venue]
			for check_in in check_ins_in_venue:
				time = self.hours_from_midnight(check_in["date"])
				h, w = self.calculate_temporal_probabilities(time)
				P_temporal_H += h
				P_temporal_W += w
			self.P_temporal_H[venue] = P_temporal_H / float(len(check_ins_in_venue))
			self.P_temporal_W[venue] = P_temporal_W / float(len(check_ins_in_venue))

		self.P_total_H = {}
		self.P_total_W = {}
		for venue in all_venues:
			self.P_total_H[venue] = self.P_temporal_H[venue] * self.P_spatial_H[venue]
			self.P_total_W[venue] = self.P_temporal_W[venue] * self.P_spatial_W[venue]
		
		self.check_ins_h = []
		self.check_ins_w = []

		for check_in in self.check_ins:
			venue = check_in["venue_id"]
			home = self.P_total_H[venue]
			work = self.P_total_W[venue]
			if home > work:
				self.check_ins_h.append(check_in)
			else:
				self.check_ins_w.append(check_in)

		if self.check_ins_h == [] or self.check_ins_w == []:
			self.parameters = None
			return


	def assign_initial_check_in_assignment(self, check_ins):
		super(StanfordModel, self).assign_initial_check_in_assignment(check_ins)
		self.parameters = {}
		self.parameters["m_h"] = np.mean([self.n_values[y] for y in [x["venue_id"] for x in self.check_ins_h]])
		self.parameters["m_w"] = np.mean([self.n_values[y] for y in [x["venue_id"] for x in self.check_ins_w]])


	def produce_max_likelihood_estimates(self, check_ins_H, check_ins_W):
		self.check_max_likelihood_estimates_input(check_ins_H, check_ins_W)
		result = {}
		home_times = [self.hours_from_midnight(x['date']) for x in check_ins_H]
		work_times = [self.hours_from_midnight(x['date']) for x in check_ins_W]
		# Temporal parameters
		result["tau_h"] = self.calculate_circular_mean(home_times)
		result["tau_w"] = self.calculate_circular_mean(work_times)
		result["sigma_h"] = self.calculate_circular_SD(home_times)
		if result["sigma_h"] < 10 ** (-4):
			result["sigma_h"] = 10 ** (-4)
		result["sigma_w"] = self.calculate_circular_SD(work_times)
		if result["sigma_w"] < 10 ** (-4):
			result["sigma_w"] = 10 ** (-4)

		return result


	def p_radiation(self, central_venue, m, venue, non_unique_venues):
		"""
		Calculate the probability that the user checks in at a particular venue.

		Returns the probability.

		central_venue -- ID of the current central venue.
		current_m -- current value of m.
		venue -- venue for which the probability is calculated.
		all_venues -- list of all venues (only Home or only Work), may contain duplicates.
		coordinates -- a dict where keys are venues and values are (latitude, longitude) 
		tuples. Usually, all venues will be in this dict.
		n_values -- n values for all venues.
		"""
		all_venues = set(non_unique_venues)
		coordinates = self.coordinates
		n_values = self.n_values
		distances = self.calculate_distances(coordinates, central_venue, all_venues)
		r_ij = distances[venue]
		venues_inside_circle = []
		for j in all_venues:
			if j == central_venue:
				continue
			if distances[j] <= r_ij:
				venues_inside_circle.append(j)
		if venue in venues_inside_circle:
			venues_inside_circle.remove(venue)
		if len(venues_inside_circle) > 0:
			s = np.sum([n_values[i] for i in venues_inside_circle])
		else:
			s = 0
		return float(m * n_values[venue]) / float((m + s) * (m + s + n_values[venue]))


	def get_m(self, current_m, central_venue, check_in_venues):
		"""
		Get m value via gradient descent optimisation.

		Returns a new m value

		current_m -- current value of m.
		central_venue -- ID of the current central venue.
		coordinates -- a dict where keys are venues and values are (latitude, longitude) 
		tuples. Usually, all venues will be in this dict.
		n_values -- n values for all venues.
		check_in_venues -- list of all venues (only Home or only Work), may contain duplicates.
		"""
		coordinates = self.coordinates
		n_values = self.n_values
		i = central_venue
		venues = set(check_in_venues)

		sum_n = 0
		for venue in venues:
			sum_n += n_values[venue]

		distances = self.calculate_distances(coordinates, i, venues)
		p = {}
		s = {}
		for j in venues:
			n = n_values[j]
			r_ij = distances[j]
			venues_inside_circle = [venue for venue in distances.keys() if distances[venue] <= r_ij]
			venues_inside_circle.remove(i)
			if j in venues_inside_circle:
				venues_inside_circle.remove(j)
			if len(venues_inside_circle) > 0:
				s[j] = np.sum([n_values[venue] for venue in venues_inside_circle])
			else:
				if i == j:
					s[j] = 0
				else:
					continue
			p[j] = float(current_m * n) / float((current_m + s[j]) * (current_m + n + s[j]))

		iterations = 100
		i = 0
		while i < iterations:
			i += 1
			old_m = current_m
			sti = 0
			for venue in venues:
				if venue not in s:
					continue
				ti = check_in_venues.count(venue)
				sti += ti * ((1.0 / (current_m + s[venue])) + (1.0 / float(current_m + n_values[venue] + s[venue])))
			current_m = float(len(check_in_venues)) / float(sti)
		return current_m


	def get_central_venue(self, all_venues, m):
		"""
		Get the central venue in the radiation model.

		Returns the ID of the central venue.

		all_venues -- list of venues, among which the central venue has to be found 
		(may contain duplicate venues).
		coordinates -- a dict where keys are venues and values are (latitude, longitude) 
		tuples. Usually, all venues will be in this dict.
		n_values -- n values for all venues.
		m -- current value of m parameter.
		"""
		venues = set(all_venues)
		coordinates = self.coordinates
		n_values = self.n_values
		#print("Total venues " + str(len(venues)))
		log_likelihoods = {}
		for i in venues:
			likelihoods_for_j = []
			distances = NCGModel.calculate_distances(coordinates, i, venues)
			p = {}
			s = {}
			for j in venues:
				n = n_values[j]
				r_ij = distances[j]
				venues_inside_circle = [venue for venue in distances.keys() if distances[venue] <= r_ij]
				venues_inside_circle.remove(i)
				if j in venues_inside_circle:
					venues_inside_circle.remove(j)
				if len(venues_inside_circle) > 0:
					s[j] = np.sum([n_values[venue] for venue in venues_inside_circle])
				else:
					if i == j:
						s[j] = 0
					else:
						continue
				p[j] = float(m * n) / float((m + s[j]) * (m + n + s[j]))
			local_likelihoods = []
			for venue in all_venues:
				if venue in s:
					likelihood = float(m * n_values[venue]) / float((m + s[venue]) * (m + n_values[venue] + s[venue]))
					local_likelihoods.append(math.log(likelihood))
			log_likelihoods[i] = np.sum(local_likelihoods)
		return max(log_likelihoods.iteritems(), key=operator.itemgetter(1))[0]


	@staticmethod
	def calculate_distances(coordinates, source, venues):
		"""
		Calculates distances from source venue to venues in venues list.

		Returns a dict where keys are venues from venues list and values are distances,
		measured in latitude-longitude value terms.

		coordinates -- a dict where keys are venues and values are (latitude, longitude) 
		tuples. Usually, all venues will be in this dict.
		source -- a venue, from which distances should be calculated.
		"""
		result = {}
		for venue in venues:
			result[venue] = NCGModel.calculate_distance(coordinates, source, venue)
		return result


	@staticmethod
	def calculate_distance(coordinates, i, j):
		"""
		Calculates a Euclidian distance between two venues.

		Returns a number representing a distance in latitude-longitude value terms.

		coordinates -- a dict where keys are venues and values are (latitude, longitude) 
		tuples. Usually, all venues will be in this dict.
		i, j -- venues, between which the distance is calculated. 
		"""
		x_i = float(coordinates[i][0])
		x_j = float(coordinates[j][0])
		y_i = float(coordinates[i][1])
		y_j = float(coordinates[j][1])
		return math.sqrt(((x_i - x_j) ** 2) + ((y_i - y_j) ** 2))


class SocialModelStanford:


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


	def get_same_day_friend_check_ins(self, date, friends, all_user_check_ins):
		friend_same_day_check_ins = []
		for friend in friends:
			if friend not in all_user_check_ins:
				continue
			friend_checkins = all_user_check_ins[friend]
			for check_in in friend_checkins:
				friend_date = check_in["date"]
				if all(getattr(date,x) == getattr(friend_date,x) for x in ['year','month','day']):
					friend_same_day_check_ins.append(check_in)
		return friend_same_day_check_ins


	def calculate_differences_for_check_ins(self, friend_same_day_check_ins, date, latitude, longitude):
		time_summed_difference = 0
		space_summed_difference = 0

		for friend_same_day_check_in in friend_same_day_check_ins:
			friend_date = friend_same_day_check_in["date"]
			time_difference = abs(friend_date - date).seconds / float(60 * 60 * 24)
			
			friend_latitude = friend_same_day_check_in["latitude"]
			latitude_difference = abs(latitude - friend_latitude)
			friend_longitude = friend_same_day_check_in["longitude"]
			longitude_difference = abs(longitude - friend_longitude)
			space_difference = math.sqrt((latitude_difference ** 2) + (longitude_difference ** 2))

			time_summed_difference += time_difference
			space_summed_difference += space_difference

		return time_summed_difference, space_summed_difference


	def get_probabilities(self, friends, all_user_check_ins, date, all_check_ins):
		result = {}
		venues = self._get_average_venue_coordinates(all_check_ins)
		for venue in venues:
			latitude = venues[venue]["latitude"]
			longitude = venues[venue]["longitude"]
			result[venue] = self.get_probability(friends, all_user_check_ins, date, latitude, longitude)
		return result


	def get_probability(self, friends, all_user_check_ins, date, latitude, longitude):
		if self.max_time_difference == self.min_time_difference or self.max_space_difference == self.min_space_difference:
			return 0.0
		
		check_ins = self.get_same_day_friend_check_ins(date, friends, all_user_check_ins)
		time_diff, space_diff = self.calculate_differences_for_check_ins(check_ins, date, latitude, longitude)
		
		if time_diff - self.min_time_difference < 0:
			prob_time = 1
		else:
			prob_time = 1 - (time_diff - self.min_time_difference) / float(self.max_time_difference - self.min_time_difference)
		
		if space_diff - self.min_space_difference < 0:
			prob_space = 1
		else:
			prob_space = 1 - (space_diff - self.min_space_difference) / float(self.max_space_difference - self.min_space_difference)

		return prob_time * prob_space


	def fit_model(self, model, friends, all_user_check_ins):

		time_summed_differences = {}
		space_summed_differences = {}

		for check_in in model.check_ins:
			latitude = check_in["latitude"]
			longitude = check_in["longitude"]
			date = check_in["date"]

			friend_same_day_check_ins = self.get_same_day_friend_check_ins(date, friends, all_user_check_ins)

			result = self.calculate_differences_for_check_ins(friend_same_day_check_ins, date, latitude, longitude)

			time_summed_differences[check_in["check_in_id"]] = result[0]
			space_summed_differences[check_in["check_in_id"]] = result[1]
			#print check_in
			#print time_difference
			#print space_difference
			#print "------"

		self.min_time_difference = np.min(time_summed_differences.values())
		self.max_time_difference = np.max(time_summed_differences.values())

		self.min_space_difference = np.min(space_summed_differences.values())
		self.max_space_difference = np.max(space_summed_differences.values())


	@staticmethod
	def calculate_entropy(a, b):
		return -1 * ((a * math.log(a, 2)) + (b * math.log(b, 2)))


class CorrectSocialModelStanford(SocialModelStanford):

	def get_probabilities(self, friends, all_user_check_ins, date, all_check_ins):
		result = {}
		venues = self._get_average_venue_coordinates(all_check_ins)
		for venue in venues:
			latitude = venues[venue]["latitude"]
			longitude = venues[venue]["longitude"]
			result[venue] = self.get_probability(friends, all_user_check_ins, date, latitude, longitude)
		return result


	def calculate_differences_for_check_ins(check_ins, date, latitude, longitude):
		None


	def get_probability(self, friends, all_user_check_ins, date, latitude, longitude):		
		friend_same_day_check_ins = self.get_same_day_friend_check_ins(date, friends, all_user_check_ins)

		sum = 0

		for check_in in friend_same_day_check_ins:
			friend_date = check_in["date"]
			time_difference = abs(friend_date - date).seconds / float(60 * 60 * 24)
			time_difference = (time_difference - self.min_time_difference) / float(self.max_time_difference - self.min_time_difference) 
				
			friend_latitude = check_in["latitude"]
			latitude_difference = abs(latitude - friend_latitude)
			friend_longitude = check_in["longitude"]
			longitude_difference = abs(longitude - friend_longitude)
			space_difference = math.sqrt((latitude_difference ** 2) + (longitude_difference ** 2))
			space_difference = (space_difference - self.min_space_difference) / float(self.max_space_difference - self.min_space_difference) 

			sum += time_difference * space_difference

		sum = (sum - self.min_sum) / float(self.max_sum - self.min_sum)

		return sum
		
		"""time_diff, space_diff = self.calculate_differences_for_check_ins(check_ins, date, latitude, longitude)
		
		if time_diff - self.min_time_difference < 0:
			prob_time = 1
		else:
			prob_time = 1 - (time_diff - self.min_time_difference) / float(self.max_time_difference - self.min_time_difference)
		
		if space_diff - self.min_space_difference < 0:
			prob_space = 1
		else:
			prob_space = 1 - (space_diff - self.min_space_difference) / float(self.max_space_difference - self.min_space_difference)

		return prob_time * prob_space"""


	def fit_model(self, model, friends, all_user_check_ins):

		time_differences = []
		space_differences = []
		sums = []

		for check_in in model.check_ins:
			latitude = check_in["latitude"]
			longitude = check_in["longitude"]
			date = check_in["date"]

			friend_same_day_check_ins = self.get_same_day_friend_check_ins(date, friends, all_user_check_ins)

			sum = 0

			for check_in in friend_same_day_check_ins:
				friend_date = check_in["date"]
				time_difference = abs(friend_date - date).seconds / float(60 * 60 * 24)
				time_differences.append(time_difference)
				
				friend_latitude = check_in["latitude"]
				latitude_difference = abs(latitude - friend_latitude)
				friend_longitude = check_in["longitude"]
				longitude_difference = abs(longitude - friend_longitude)
				space_difference = math.sqrt((latitude_difference ** 2) + (longitude_difference ** 2))
				space_differences.append(space_difference)

				sum += time_difference * space_difference

			sums.append(sum)
			
		self.min_time_difference = np.min(time_differences)
		self.max_time_difference = np.max(time_differences)
		if self.min_time_difference == self.max_time_difference:
			self.min_time_difference = 0

		self.min_space_difference = np.min(space_differences)
		self.max_space_difference = np.max(space_differences)
		if self.min_space_difference == self.max_space_difference:
			self.min_space_difference = 0

		self.min_sum = np.min(sums)
		self.max_sum = np.max(sums)
		if self.min_sum == self.max_sum:
			self.min_sum = 0

		#print self.min_time_difference
		#print self.max_time_difference
		#print self.min_space_difference
		#print self.max_space_difference
		#print self.min_sum
		#print self.max_sum


class AdvancedSocialModel(SocialModelStanford):

	def __init__(self, all_user_check_ins, network, current_user):
		G = nx.DiGraph()
		for user in all_user_check_ins:
			for check_in in all_user_check_ins[user]:
				venue = check_in["venue_id"]
				if user not in G.nodes():
					G.add_node(user)
				if venue not in G.nodes():
					G.add_node(user)
				if (user, venue) not in G.edges():
					G.add_edge(user, venue, weight = 1)
				else:
					current_weight = G.get_edge_data(user, venue)['weight']
					G.add_edge(user, venue, weight = current_weight + 1)
		(hub_scores, authority_scores) = nx.hits(G)
		self.authority_scores = authority_scores
		self.user = current_user
		self.user_check_ins = all_user_check_ins[current_user]
		self.network = network

		friend_count = {}
		for user in network:
			friend_count[user] = len(network[user])

		self.friend_count = friend_count


	def get_probabilities(self, friends, all_user_check_ins, date, all_check_ins):
		result = {}
		venues = self._get_average_venue_coordinates(all_check_ins)
		for venue in venues:
			latitude = venues[venue]["latitude"]
			longitude = venues[venue]["longitude"]
			result[venue] = self.get_probability(friends, all_user_check_ins, date, latitude, longitude)
		return result


	def get_probability(self, friends, all_user_check_ins, date, latitude, longitude):
		friends_influences = {}
		for friend in self.network[self.user]:
			friends_influences[friend] = self.friend_count[friend]
		for friend in friends_influences:
			friends_influences[friend] = friends_influences[friend] / float(np.max(friends_influences.values()))
		friend_influence_multipliers = []
		for friend in friends:
			if friend not in all_user_check_ins:
				continue
			for check_in in all_user_check_ins[friend]:
				friend_influence_multipliers.append(friends_influences[friend])
		friend_influence_multiplier = np.mean(friend_influence_multipliers)

		if self.max_time_difference == self.min_time_difference or self.max_space_difference == self.min_space_difference:
			return 0.0
		
		check_ins = self.get_same_day_friend_check_ins(date, friends, all_user_check_ins)
		time_diff, space_diff = self.calculate_differences_for_check_ins(check_ins, date, latitude, longitude)
		
		if time_diff - self.min_time_difference < 0:
			prob_time = 1
		else:
			prob_time = 1 - (time_diff - self.min_time_difference) / float(self.max_time_difference - self.min_time_difference)
		
		if space_diff - self.min_space_difference < 0:
			prob_space = 1
		else:
			prob_space = 1 - (space_diff - self.min_space_difference) / float(self.max_space_difference - self.min_space_difference)

		return prob_time * prob_space * friend_influence_multiplier


class SimpleSocialModel(SocialModelStanford):

	def __init__(self, all_user_check_ins, friends, user):
		self.all_user_check_ins = all_user_check_ins
		self.friends = friends
		self.user = user


	def get_probabilities(self, friends, all_user_check_ins, date, all_check_ins):
		result = {}
		venues = self._get_average_venue_coordinates(all_check_ins)
		for venue in venues:
			result[venue] = self.count_social_checkins(venue, date)
		for venue in result:
			result[venue] /= float(np.max(result.values()))
		return result


	def count_social_checkins(self, venue, date):
		social_check_ins = 0
		for friend in self.all_user_check_ins:
			for check_in in self.all_user_check_ins[friend]:
				time_distance = abs(date - check_in["date"]).seconds / 60.0 / 60.0
				if check_in["venue_id"] == venue and time_distance <= 2:
					social_check_ins += 1
		return social_check_ins


if __name__ == '__main__':
	check_ins = [{'venue_id': '1', 'latitude': 60, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 220, 'date': datetime.datetime(2012, 7, 18, 15, 30, 00)},
                 {'venue_id': '2', 'latitude': 42, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 22, 'date': datetime.datetime(2012, 7, 18, 15, 15, 00)},
                 {'venue_id': '2', 'latitude': 70, 'check_in_message': 'empty_message', 'check_in_id': '141', 'longitude': 210, 'date': datetime.datetime(2012, 7, 18, 15, 45, 00)}]
	model = SocialModelStanford()
	result = model.calculate_differences_for_check_ins(check_ins, datetime.datetime(2012, 7, 18, 16, 00, 00), 62, 215)
	print result