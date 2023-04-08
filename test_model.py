import argparse
import random
from pathlib import Path

# Extern
import matplotlib.pyplot as plt
import numpy as np
import torch


# Custom
import model
import dataset
from sklearn.metrics import multilabel_confusion_matrix
import common


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--device', type=str, default='cpu', help='cpu or 0, 1, ...')
	parser.add_argument('-d', '--data', type=str, help='file of dataset')
	parser.add_argument('-w', '--weights', type=str, help='weitghts of model')
	parser.add_argument('--a', type=float, default=0.5, help='s cut parameter')
	parser.add_argument('--image-size', type=int, default=240, help='input and target image size')
	args = parser.parse_args()

	cfg = vars(args)

	args = argparse.Namespace(**cfg)
	print(args, '\n')

	assert args.data is not None
	return args

def f_macro(matrix):
	print(matrix)
	matrix = matrix.sum(0)
	pr = 0
	if (matrix[1,1]+matrix[0,1]) != 0:
		pr = matrix[1,1]/(matrix[1,1]+matrix[0,1])
	rc = matrix[1,1]/(matrix[1,1]+matrix[1,0])

	return 2*pr*rc/(pr+rc)

def f_micro(matrix):
	pr = 0
	rc = 0
	for i in range(matrix.shape[0]):
		if (matrix[i, 1,1]+matrix[i, 0,1]) != 0:
			pr += matrix[i, 1,1]/(matrix[i, 1,1]+matrix[i, 0,1])
		rc += matrix[i, 1,1]/(matrix[i, 1,1]+matrix[i, 1,0])
	pr /= matrix.shape[0]
	rc /= matrix.shape[0]
	return 2*pr*rc/(pr+rc)
def my_f1(matrix):
	f = 0
	for i in range(matrix.shape[0]):
		pr = 0
		rc = 0
		if (matrix[i, 1,1]+matrix[i, 0,1]) != 0:
			pr = matrix[i, 1,1]/(matrix[i, 1,1]+matrix[i, 0,1])
		rc = matrix[i, 1,1]/(matrix[i, 1,1]+matrix[i, 1,0])
		f += 2*pr*rc/(pr+rc)
	return f/matrix.shape[0]


if __name__ == "__main__":
	args = parse_args()

	device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'
	if device != 'cpu':
		assert torch.cuda.is_available()

	model = model.get_model()
	model.load_state_dict(torch.load(args.weights)['state_dict'])
	model.eval()
	model.to(device)
	
	ds = dataset.get_test_dataloader(args)
	
	table = None
	a = args.a
	
	table = np.zeros((4, 2, 2)) # [TN, FP]
                                # [FN, TP]
	
	
	for i in range(ds.__len__()):
		print(i)
		batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh, batch_t, batch_c = ds.__getitem__(i)
		batch_i_hh = batch_i_hh.unsqueeze(dim=0).to(device)
		batch_i_hv = batch_i_hv.unsqueeze(dim=0).to(device)
		batch_a_hh = batch_a_hh.unsqueeze(dim=0).to(device)
		batch_a_hv = batch_a_hv.unsqueeze(dim=0).to(device)
		batch_a_hh_hv = batch_a_hh_hv.unsqueeze(dim=0).to(device)
		batch_a_hv_hh = batch_a_hv_hh.unsqueeze(dim=0).to(device)
		classes = batch_c.squeeze()
		preds = model(batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh).squeeze()
		for i in range(preds.shape[0]):
			if classes[i] == 1:
				if preds[i] > a:
					table[i, 1,1] += 1
				else:
					table[i, 1,0] += 1
			else:
				if preds[i] > a:
					table[i, 0,1] += 1
				else:
					table[i, 0,0] += 1
                
	metrics = my_f1(table)
	macro = f_macro(table)
	micro = f_micro(table)
	print(metrics, macro, micro)

"""
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params,  sum([p.numel() for p in model.parameters() if p.requires_grad]))
print(model(torch.zeros((5, 1, 64, 64)), torch.zeros((5, 1, 64, 64)), torch.zeros((5, 1, 33, 33)), torch.zeros((5, 1, 33, 33)), 
	torch.zeros((5, 1, 33, 33)), torch.zeros((5, 1, 33, 33))) )
"""
