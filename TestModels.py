import datetimeimport unittestfrom Models import Modelclass TestModel(unittest.TestCase):    def setUp(self):        self.model = Model()    def test_produce_initial_check_in_assignment_correct_clustering(self):        check_ins = [                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '15', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}        ]        cluster1, cluster2 = self.model.produce_initial_check_in_assignment(check_ins)        expected_cluster1 = [{'venue_id': '41059b00f964a520850b1fe3', 'date': datetime.datetime(2012, 7, 18, 14, 43, 38), 'longitude': 122.386, 'check_in_id': '14', 'check_in_message': 'empty_message', 'latitude': 50.6164}, {'venue_id': '41059b00f964a520850b1fe3', 'date': datetime.datetime(2012, 7, 18, 14, 43, 38), 'longitude': 120.386, 'check_in_id': '15', 'check_in_message': 'empty_message', 'latitude': 51}]        expected_cluster2 = [{'venue_id': '41059b00f964a520850b1fe3', 'date': datetime.datetime(2012, 7, 18, 14, 43, 38), 'longitude': -122.386, 'check_in_id': '12', 'check_in_message': 'empty_message', 'latitude': 37.6164}, {'venue_id': '41059b00f964a520850b1fe3', 'date': datetime.datetime(2012, 7, 18, 14, 43, 38), 'longitude': -120.386, 'check_in_id': '13', 'check_in_message': 'empty_message', 'latitude': 35}]        for i in range(0, len(cluster1)):            self.assertDictEqual(cluster1[i], expected_cluster1[i])        for i in range(0, len(cluster2)):            self.assertDictEqual(cluster2[i], expected_cluster2[i])    def test_produce_initial_check_in_assignment_duplicate_ids(self):        check_ins = [                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}        ]        try:            self.model.produce_initial_check_in_assignment(check_ins)            self.fail("No error produced when several check-ins have same IDs")        except ValueError as e:            expected = "Error: some check-ins have same IDs!"            actual = str(e)            self.assertEqual(expected, actual)        except:            self.fail("Wrong exception was produced")    def test_produce_initial_check_in_assignment_no_id(self):        check_ins = [                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 50.6164, 'check_in_message': 'empty_message', 'longitude': 122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}        ]        try:            self.model.produce_initial_check_in_assignment(check_ins)            self.fail("No error produced when a check-in does not have an ID")        except ValueError as e:            expected = "Error: one of check-ins does not have ID!"            actual = str(e)            self.assertEqual(expected, actual)        except:            self.fail("Wrong exception was produced")    def test_produce_initial_check_in_assignment_no_latitude(self):        check_ins = [                {'venue_id': '41059b00f964a520850b1fe3', 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': -120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}        ]        try:            self.model.produce_initial_check_in_assignment(check_ins)            self.fail("No error produced when a check-in does not have latitude")        except ValueError as e:            expected = "Error: check-in 12 does not have latitude!"            actual = str(e)            self.assertEqual(expected, actual)        except:            self.fail("Wrong exception was produced")    def test_produce_initial_check_in_assignment_no_longitude(self):        check_ins = [                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 51, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': 120.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},                {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 35, 'check_in_message': 'empty_message', 'check_in_id': '13', 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}        ]        try:            self.model.produce_initial_check_in_assignment(check_ins)            self.fail("No error produced when a check-in does not have longitude")        except ValueError as e:            expected = "Error: check-in 13 does not have longitude!"            actual = str(e)            self.assertEqual(expected, actual)        except:            self.fail("Wrong exception was produced")if __name__ == '__main__':    unittest.main()