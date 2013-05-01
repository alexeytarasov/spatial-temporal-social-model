__author__ = 'alexeytarasov'

import cStringIO
import datetime
import unittest

from DataLoader import DataLoader
from mocker import Mocker

class TestDataLoader(unittest.TestCase):


    def setUp(self):
        self.mocker = Mocker()
        self.file = cStringIO.StringIO()
        self.file2 = cStringIO.StringIO()


    def tearDown(self):
        self.mocker.restore()
        self.mocker.verify()
        self.file.close()
        self.file2.close()


    def test_single_file_happy_path(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        expected = {'418': [{'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}, {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)}]}
        actual = DataLoader.load_check_ins_from_file(self.file)
        self.assertDictEqual(expected, actual)


    def test_invalid_number_of_check_in_parameters(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, "Error in line 2: the line should contain user_id, check-in_id, date, latitude, longitude, venue_id and check-in_message, separated by |")
        

    def test_empty_strings_in_middle(self):
        self.file.write("\n418|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, "Error in line 1: the line should contain user_id, check-in_id, date, latitude, longitude, venue_id and check-in_message, separated by |")
            
       
    def test_empty_strings_in_end(self):
        self.file.write("418|23|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n ")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, "Error in line 2: the line should contain user_id, check-in_id, date, latitude, longitude, venue_id and check-in_message, separated by |")
        

    def test_invalid_date(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|123asd|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: invalid format of date, should be YYYY-MM-DD HH:MM:SS')
        

    def test_longitude_not_a_number(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|45.54|a|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: longitude should be a float number')
        

    def test_longitude_out_of_bounds(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|45.5|-190.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: longitude should be between -90 and 90')
        

    def test_latitude_not_a_number(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|abcd|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: latitude should be a float number')
        

    def test_latitude_out_of_bounds(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|100|-122.386|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: latitude should be between -90 and 90')
        

    def test_invalid_venue(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|34|-122.386||empty_message")
        self.file.seek(0)
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_file(self.file)
        self.assertEqual(cm.exception.message, 'Error in line 2: venue_id can not be an empty string')
        

    def test_single_directory_happy_path(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|13|2012-07-18 12:34:45|45.54|45.6|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        self.file2.write("418|14|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|15|2012-07-18 12:34:45|45.54|45.6|41059b00f964a520850b1fe3|empty_message")
        self.file2.seek(0)

        mock_glob = self.mocker.replace('glob.glob')
        mock_glob("some_directory")
        self.mocker.result(['.', 'file1', 'file2'])

        mock_open = self.mocker.replace('__builtin__.open')
        mock_open("file1")
        self.mocker.result(self.file)
        mock_open("file2")
        self.mocker.result(self.file2)

        self.mocker.replay()
        expected_dict = {
            '418': [{'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '12', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
                    {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 45.54, 'check_in_message': 'empty_message', 'check_in_id': '13', 'longitude': 45.6, 'date': datetime.datetime(2012, 7, 18, 12, 34, 45)},
                    {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 37.6164, 'check_in_message': 'empty_message', 'check_in_id': '14', 'longitude': -122.386, 'date': datetime.datetime(2012, 7, 18, 14, 43, 38)},
                    {'venue_id': '41059b00f964a520850b1fe3', 'latitude': 45.54, 'check_in_message': 'empty_message', 'check_in_id': '15', 'longitude': 45.6, 'date': datetime.datetime(2012, 7, 18, 12, 34, 45)}]}
        actual_dict = DataLoader.load_check_ins_from_directory("some_directory")
        self.assertDictEqual(expected_dict, actual_dict)


    def test_same_check_in_ids_in_different_files(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|13|2012-07-18 12:34:45|45.54|45.6|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)
        self.file2.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|15|2012-07-18 12:34:45|45.54|45.6|41059b00f964a520850b1fe3|empty_message")
        self.file2.seek(0)

        mock_glob = self.mocker.replace('glob.glob')
        mock_glob("some_directory")
        self.mocker.result(['.', 'file1', 'file2'])

        mock_open = self.mocker.replace('__builtin__.open')
        mock_open("file1")
        self.mocker.result(self.file)
        mock_open("file2")
        self.mocker.result(self.file2)

        self.mocker.replay()
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_directory("some_directory")
        self.assertEqual(cm.exception.message, 'Error processing file file2: check-in with ID 12 has already been encountered for user 418')
        

    def test_same_check_in_ids_in_same_file(self):
        self.file.write("418|12|2012-07-18 14:43:38|37.6164|-122.386|41059b00f964a520850b1fe3|empty_message\n418|12|2012-07-18 12:34:45|45.54|45.6|41059b00f964a520850b1fe3|empty_message")
        self.file.seek(0)

        mock_glob = self.mocker.replace('glob.glob')
        mock_glob("some_directory")
        self.mocker.result(['.', 'file1'])

        mock_open = self.mocker.replace('__builtin__.open')
        mock_open("file1")
        self.mocker.result(self.file)

        self.mocker.replay()
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_directory("some_directory")
        self.assertEqual(cm.exception.message, 'Error processing file file1: check-in with ID 12 has already been encountered for user 418')
        

    def test_empty_directory(self):
        mock_glob = self.mocker.replace('glob.glob')
        mock_glob("some_directory")
        self.mocker.result(['.'])
        self.mocker.replay()
        with self.assertRaises(ValueError) as cm:
            DataLoader.load_check_ins_from_directory("some_directory")
        self.assertEqual(cm.exception.message, 'Error: directory some_directory is empty')


if __name__ == '__main__':
    unittest.main()