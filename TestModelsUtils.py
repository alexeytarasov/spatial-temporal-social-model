import datetime
import unittest

from Models import Model

class TestModel(unittest.TestCase):


	def setUp(self):
		self.model = Model()


	def test_produce_initial_max_likelihood_estimates_invalid_check_ins(self):
	    check_ins_invalid = [
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}
	    ]
	    check_ins_valid = [
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '15', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}
	    ]
	    with self.assertRaises(ValueError) as cm:
	        self.model.check_max_likelihood_estimates_input(check_ins_invalid, check_ins_valid)
	    self.assertEqual(cm.exception.message, "Error: some check-ins have same IDs!")


	def test_produce_initial_check_in_assignment_invalid_check_ins(self):
		check_ins_invalid = [
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
	            {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}
	    ]
		with self.assertRaises(ValueError) as cm:
			self.model.produce_initial_check_in_assignment(check_ins_invalid)
		self.assertEqual(cm.exception.message, "Error: some check-ins have same IDs!")