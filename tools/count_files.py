import argparse
import os

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, help='file of dataset')
	args = parser.parse_args()
	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print(args, '\n')

	return args

if __name__ == "__main__":

	args = parse_args()
	print(len(os.listdir(args.data)))
