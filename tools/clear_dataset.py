import os
import numpy as np
from PIL import Image
import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-d', '--data', type=str, help='clear path')
	args = parser.parse_args()

	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print(args, '\n')
	return args

if __name__ == '__main__':

	args = parse_args()

	for path in ["input_hh", "input_hv", "autocors_hh", "autocors_hv", "autocors_hh_hv", "autocors_hv_hh", "classes", "target"]:
		full_path = os.path.join(args.data, path)

		for file in os.listdir(full_path):
			name = os.path.join(full_path, file)
			os.remove(name)