import cStringIO

__author__ = 'alexeytarasov'

from datetime import datetime
import glob
import csv

from Utils import *


class DataLoader:


    @staticmethod
    def load_social_network(file):
        results = {}
        reader = csv.reader(file, delimiter='|')
        current_line = 0
        for connection in reader:
            user_a = connection[0]
            user_b = connection[1]
            if user_a not in results:
                results[user_a] = []
            if user_b not in results:
                results[user_b] = []
            results[user_a].append(user_b)
            results[user_b].append(user_a)
        return results


    @staticmethod
    def load_check_ins_from_directory(directory_name):
        """
        Loads check-ins from csv files located in a particular directory. Method load_check_ins_from_file is used to
        process each file.

        Returns a dicts with a key being a user name, and a value being a list of check-ins.

        directory_name -- name of the directory containing files from which to extract check-ins.
        """
        results = {}
        file_names = glob.glob(directory_name + "/*")
        file_names = [x for x in file_names if not x.startswith(".")]
        if len(file_names) == 0:
            raise ValueError("Error: directory {directory} is empty".format(directory=directory_name))
        for file_name in file_names:
            result = DataLoader.load_check_ins_from_file(open(file_name, 'rU'))
            for user in result:
                for check_in in result[user]:
                    id = check_in["check_in_id"]
                    for previous_user in results:
                        if id in [x["check_in_id"] for x in results[previous_user]]:
                            raise ValueError("Error processing file {file_name}: check-in with ID {check_in_id} has already been encountered for user {user}".format(file_name=file_name,check_in_id=id,user=previous_user))
                    if user not in results:
                        results[user] = []
                    results[user].append(check_in)
        return results


    @staticmethod
    def load_check_ins_from_file(file):
        """
        Loads check-ins from a csv file. Each check-in uses | as a delimiter and has the following format:
        user_id|check_in_id|YYYY-MM-DD hh:mm:ss|latitude|longitude|venue_id|check_in_message

        Returns a dict with a key being a user name, and a value being a list of check-ins. Each item in the list is a
        dict, containing all information about the check-in.

        file -- file object from which to extract check-ins.
        """
        results = {}
        reader = csv.reader(file, delimiter='|')
        current_line = 0
        for check_in in reader:
            current_line += 1
            if len(check_in) != 7:
                raise ValueError('Error in line {current_line}: the line should contain user_id, check-in_id, date, latitude, longitude, venue_id and check-in_message, separated by |'.format(current_line=current_line))
            user_id = check_in[0]
            if user_id not in results:
                results[user_id] = []
            result = {}
            result['check_in_id'] = check_in[1]
            #---------------------------------------------------------
            try:
                result['date'] = datetime.strptime(check_in[2], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError('Error in line {current_line}: invalid format of date, should be YYYY-MM-DD HH:MM:SS'.format(current_line=current_line))
            #---------------------------------------------------------
            try:
                result['latitude'] = float(check_in[3])
            except ValueError:
                raise ValueError('Error in line {current_line}: latitude should be a float number'.format(current_line=current_line))
            if result['latitude'] < -90 or result['latitude'] > 90:
                raise ValueError('Error in line {current_line}: latitude should be between -90 and 90'.format(current_line=current_line))
            #---------------------------------------------------------
            try:
                result['longitude'] = float(check_in[4])
            except ValueError:
                raise ValueError('Error in line {current_line}: longitude should be a float number'.format(current_line=current_line))
            if result['longitude'] < -180 or result['longitude'] > 180:
                raise ValueError('Error in line {current_line}: longitude should be between -90 and 90'.format(current_line=current_line))
            #---------------------------------------------------------
            result['venue_id'] = check_in[5]
            if result['venue_id'] == '':
                raise ValueError('Error in line {current_line}: venue_id can not be an empty string'.format(current_line=current_line))
            #---------------------------------------------------------
            result['check_in_message'] = check_in[6]
            results[user_id].append(result)
        return results