from copy import deepcopy
import json
import numpy as np

class input_dict(dict):
	"""input dictionary class
		-> with dot.notation access to dictionary attributes"""

	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __new__(cls, *args, **kwargs):
		obj = dict().__new__(cls)
		return obj

	def __init__(self, fname=None):
		if(fname is not None):
			if(isinstance(fname, str)):       
				with open(fname, 'r') as f:
					d = json.load(f)
			elif(isinstance(fname, dict)):
				d = fname
			self.update(d)
	
			for k in self.keys():
				if(isinstance(self[k], list)):
					self[k] = np.array(self[k])

	def set_if_none(self, key, val):
		if(key not in self.keys()):
			self[key] = val

	def has_key(self, key):
		if(key in self.keys()):
			return True
		else:
			return False

	def copy(self, memo=None):
		return input_dict(deepcopy(dict(self), memo=memo))

	def save_json(self, fname):
		def default(obj):
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			else:
				return None
			# raise TypeError(f'Not serializable {obj}')

		with open(fname, 'w') as f:
			json.dump(self, f, default=default, indent=4, sort_keys=True)

	def get_keys(self):
		return list(self.keys())

	def __dir__(self):
		return dict().__dir__() + self.get_keys() + ['get_keys','save_json','copy','has_key','set_if_none']