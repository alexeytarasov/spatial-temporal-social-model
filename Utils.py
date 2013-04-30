class Utils:

	@staticmethod
	def check_check_in_syntax(check_in):
		"""
		Tests if check-in has all required attributes and they have valid values.

		Returns True if everything is OK, throws an exception otherwise.

		check_in -- a dict containing information about check-in. Should have at least
		venue_id, latitude, longitude, check_in_id and date keys. May contain any additional
		attributes as well. 
		"""
		if not isinstance(check_in, dict):
			raise ValueError("Check-in should be a dictionary!")
		check_in_attributes = check_in.keys()

		if 'check_in_id' not in check_in_attributes:
			raise ValueError("Check-in should contain check_in_id!")

		if 'venue_id' not in check_in_attributes:
			raise ValueError("Check-in should contain venue_id!")

		if 'latitude' not in check_in_attributes:
			raise ValueError("Error: check-in {id} does not have latitude!".format(id=check_in["check_in_id"]))
		try:
			float(check_in['latitude'])
		except (TypeError, ValueError):
			raise ValueError("Check-in has invalid latitude!")

		if 'longitude' not in check_in_attributes:
			raise ValueError("Error: check-in {id} does not have longitude!".format(id=check_in["check_in_id"]))
		try:
			float(check_in['longitude'])
		except (TypeError, ValueError):
			raise ValueError("Check-in has invalid longitude!")

		if 'date' not in check_in_attributes:
			raise ValueError("Check-in should contain date!")

		return True

	@staticmethod
	def check_userless_check_in_list(check_ins):
		if not isinstance(check_ins, list):
			raise ValueError("Error: the input argument is not a valid list!")
		if len(check_ins) == 0:
			raise ValueError("Error: the list of check-ins is empty!")
		if len(check_ins) == 1:
			raise ValueError("Error: the list should contain at least two check-ins!")
		for check_in in check_ins:
			Utils.check_check_in_syntax(check_in)
		ids = [x["check_in_id"] for x in check_ins]
		if len(ids) != len(set(ids)):
			raise ValueError("Error: some check-ins have same IDs!")