import os
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-d', '--data', type=str, help='file of dataset')
	args = parser.parse_args()

	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print(args, '\n')
	return args

if __name__ == '__main__':
	args = parse_args()

	for name in os.listdir(os.path.join(args.data, "target")):
        
		t = np.array(Image.open(os.path.join(args.data, "target", name)))
		cols = np.unique(t.reshape(-1, t.shape[2]), axis=0)
		for i in cols:
			if int(np.sum(i).item()) == 0:
				print(name)
				os.remove(os.path.join(args.data, "target", name))
				os.remove(os.path.join(args.data, "input_hh", name))
				os.remove(os.path.join(args.data, "input_hv", name))
				os.remove(os.path.join(args.data, "autocors_hh", name))
				os.remove(os.path.join(args.data, "autocors_hv", name))
				os.remove(os.path.join(args.data, "autocors_hh_hv", name))
				os.remove(os.path.join(args.data, "autocors_hv_hh", name))
				os.remove(os.path.join(args.data, "classes", name+'.npy'))
				break




