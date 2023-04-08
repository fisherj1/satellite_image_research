import argparse

# Extern
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path
# Custom
import model
import dataset
from sklearn.metrics import f1_score
import common

from torch.utils.tensorboard import SummaryWriter


def parse_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--device', type=str, default='cpu', help='cpu or 0, 1, ...')
	parser.add_argument('-d', '--data', type=str, help='path of test data')
	parser.add_argument('-w', '--weights-dir', type=str, help='path to weights for loading (including file name)')
	parser.add_argument('--f-score', action='store_true', help='using f-score metrics')
	parser.add_argument('--a', type=float, default=0.5, help='s-cut porog')
	parser.add_argument('--image-size', type=int, default=240, help='input and target image size')
	
	args = parser.parse_args()
	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print('\n', args, '\n')
	
	assert args.data is not None
	assert args.weights_dir is not None
	return args


def visualizate(model, dataset, cfg):
	h = 7
	w = dataset.__len__()
	print(w, h)
	names = ['hh', 'hv', 'autocor_hh', 'autocor_hv', 'autocor_hh_hv', 'autocor_hv_hh', 'target']

	metrics = None
	if cfg.f_score:
		metrics = f1_score
	figs = []
	num = 0
	for fig_num in range(w//6):
		fig = plt.figure(figsize=(6*5, h*5))
		for i in range(6):
			k = i+1
			item = dataset.__getitem__(num)
			num += 1
			for j in range(h):
				ax = fig.add_subplot(h, w, k)
				if j != h-1:
					ax.imshow(item[j].numpy().transpose(1,2,0), cmap='gray')
				else:
					ax.imshow(item[j].numpy().transpose(1,2,0))
				ax.set_ylabel(names[j])
				if j == 0:
					y = item[-1]
					y_pred = model(item[0].unsqueeze(dim=0), item[1].unsqueeze(dim=0), item[2].unsqueeze(dim=0), item[3].unsqueeze(dim=0), item[4].unsqueeze(dim=0), item[5].unsqueeze(dim=0))
					y_pred = (y_pred>cfg.a).float()
					title = ''
					#print(y_pred.shape, y.shape)
					if metrics is not None:
						f = 'f_score: ' + str(metrics(y.float().squeeze().numpy(), y_pred[0].float().squeeze().numpy())) + '\n'
						title += f
					title += 'classes: ' + str(y.squeeze().numpy().tolist())+'\n'
					title += 'preds: ' + str(y_pred.squeeze().numpy().tolist()) + ' with a=' + str(args.a)
					ax.set_title(title)
				k += 6
		figs.append(fig)
	return figs

if __name__ == "__main__":
	args = parse_args()

	# Setup device
	device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'
	if device != 'cpu':
		assert torch.cuda.is_available()
	
	# Get model
	model = model.get_model()
	
	# Load weights and get ready for use
	model.load_state_dict(torch.load(args.weights_dir)['state_dict'], strict=False)
	model.eval()

	dataset = dataset.get_test_dataloader(args)

	figs = visualizate(model, dataset, args)
	name = args.weights_dir.split('/')[-3]
	test_dir = common.increment_path(Path('runs/test') / name, exist_ok=True, mkdir=True)
	print(str(test_dir))
	tb_logger = SummaryWriter(str(test_dir))

	for i in figs:
		i.savefig('fig.png')
		tb_logger.add_figure('test',i, global_step=len(figs))
