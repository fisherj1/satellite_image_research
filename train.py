#lol
# Built-in
import argparse
import random
import sys

from pathlib import Path

# Extern
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy as BCE
import torchvision
import yaml

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

# Custom
import model
import dataset
import common


def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	# For cuda
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--config', type=str, default=None,
						help='path to config file, other args will be ingored ')

	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument('--device', type=str, default='cpu', help='cpu or 0, 1, ...')
	parser.add_argument('-d', '--data', type=str, help='file of dataset')
	parser.add_argument('--project', default='runs/train', help='save to project/name')
	parser.add_argument('--name', default='exp', help='save to project/name')
	parser.add_argument('--exist-ok', action='store_true', 
						 help='existing project/name ok, do not increment and overwrite existing files')

	parser.add_argument('--train-bs', type=int, default=8, help='train batch size')
	parser.add_argument('--valid-bs', type=int, default=32, help='valid batch size')
	parser.add_argument('-e', '--epochs', type=int, help='path to experiment config')
	parser.add_argument('--lr', type=float, default=0.00001, help='base learning rate')
	parser.add_argument('--image-size', type=int, default=240, help='input and target image size')
	parser.add_argument('--norm-to-max', type=float, default=None, help='noorm to max')
	parser.add_argument('--mse', action='store_true', help='using mse loss')
	parser.add_argument('--f-score', action='store_true', default=False, help='using f-score metrics')
	parser.add_argument('--a', type=float, default=0.5, help='s-cut parameter')


	args = parser.parse_args()

	if args.config is not None:
		dict_args = vars(args)
		with open(dict_args.pop('config'), 'r') as f:
			cfg = yaml.safe_load(f)
	else:
		cfg = vars(args)
		del cfg['config']

	args = argparse.Namespace(**cfg)
	print(args, '\n')

	assert args.data is not None
	assert args.epochs is not None

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



def train_one_epoch(model, opt, data_loader,
					criterion, device, metrics, a):
	print('a=', a)

	info = {'loss': 0,
			'item_loss': [], 
			'metrics' : 0,
			'item_metrics' : [],
			'table' : None,
			'macro_metrics' : 0,
			'micro_metrics' : 0,
			}
	samples_amount = 0

	model.train()
	for i, (batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh, batch_t, batch_c) in tqdm(enumerate(data_loader), total=len(data_loader)):
		batch_len = len(batch_t)

		opt.zero_grad()

		batch_i_hh = batch_i_hh.to(device)
		batch_i_hv = batch_i_hv.to(device)
		batch_a_hh = batch_a_hh.to(device)
		batch_a_hv = batch_a_hv.to(device)
		batch_a_hh_hv = batch_a_hh_hv.to(device)
		batch_a_hv_hh = batch_a_hv_hh.to(device)
		#print(batch_a_hv.shape)

		#target_imgs = batch_t.to(device)
		classes = batch_c.squeeze(1).squeeze(-1).to(device)
		# Model inference'
		preds = model(batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh)
		batch_loss = criterion(preds.float(), classes.float(), reduction='none')
		if batch_loss.dim() > 1:
			batch_loss = batch_loss.mean(1)
		#print(batch_loss)
		mean_loss = torch.mean(batch_loss)
		info['loss'] += mean_loss.item()*batch_len
		info['item_loss'] += batch_loss.detach().cpu().numpy().tolist()
		mean_loss.backward()
		opt.step()
		samples_amount += batch_len

		
		#print((preds[i]>a).float().detach().cpu().numpy(), preds)
		if metrics != None:
			for i in range(batch_len):
				info['item_metrics'].append(metrics((classes[i]>0).float().detach().cpu().numpy(), (preds[i]>a).float().detach().cpu().numpy()).item())
			info['metrics'] += sum(info['item_metrics'][-batch_len:])

			if info['table'] is None:
				info['table'] = multilabel_confusion_matrix((classes>0).float().detach().cpu().numpy(), (preds>a).float().detach().cpu().numpy())
			else:
				info['table'] += multilabel_confusion_matrix((classes>0).float().detach().cpu().numpy(), (preds>a).float().detach().cpu().numpy())

	info['metrics'] = my_f1(info['table']) #/= samples_amount	
	info['macro_metrics'] = f_macro(info['table'])
	info['micro_metrics'] = f_micro(info['table'])

	info['loss'] /= samples_amount
	
	

	return info


@torch.no_grad()
def valid_one_epoch(model, data_loader,
					criterion, device, metrics, a):

	info = {'loss': 0,
			'item_loss': [] , 
			'preds': [], 
			'metrics' : 0,
			'item_metrics' : [],
			'table' : None,
			'macro_metrics' : 0,
			'micro_metrics' : 0,
}

	best = {'inp': None,
			'out': None,
			'loss': None,
			'target': None,
			'metrics': None}
	
	worst = {'inp': None,
			'target': None,
			'loss': None,
			'target': None,
			'metrics': None}

	com = {'inp': None, 'out': None, 'loss': None, 'target': None, 'metrics': None}
	
	k = False
	samples_amount = 0

	model.eval()
	for i, (batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh, batch_t, batch_c) in tqdm(enumerate(data_loader), total=len(data_loader)):
		batch_len = len(batch_t)

		batch_i_hh = batch_i_hh.to(device)
		batch_i_hv = batch_i_hv.to(device)
		batch_a_hh = batch_a_hh.to(device)
		batch_a_hv = batch_a_hv.to(device)
		batch_a_hh_hv = batch_a_hh_hv.to(device)
		batch_a_hv_hh = batch_a_hv_hh.to(device)

		#target_imgs = batch_t.to(device)
		classes = batch_c.squeeze(1).squeeze(-1).to(device)
		# Model inference'
		preds = model(batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh)
		#print(preds.shape, classes.shape)
		loss = criterion(preds.float(), classes.float(), reduction='none')
		if loss.dim() > 1:
			loss = loss.mean(1)

		#print(loss.shape)
		mean_loss = torch.mean(loss)
		if metrics != None:
			for i in range(batch_len):
				info['item_metrics'].append(metrics((classes[i]>0).float().detach().cpu().numpy(), (preds[i]>a).float().detach().cpu().numpy()).item())
			info['metrics'] += sum(info['item_metrics'][-batch_len:])

			if info['table'] is None:
				info['table'] = multilabel_confusion_matrix((classes>0).float().detach().cpu().numpy(), (preds>a).float().detach().cpu().numpy())
			else:
				info['table'] += multilabel_confusion_matrix((classes>0).float().detach().cpu().numpy(), (preds>a).float().detach().cpu().numpy())

		#Вывод картинок на валидации#################################3		
		"""
		input_list = [batch_i_hh, batch_i_hv, batch_a_hh, batch_a_hv, batch_a_hh_hv, batch_a_hv_hh, batch_t, batch_c]

		min_loss = torch.min(loss)
		if best['loss'] is None or best['loss'] > min_loss:
			ind = torch.argmin(loss)
			#print(ind, loss)
			best['loss'] = min_loss.item()
			best['inp']  = [i[ind].detach().cpu().numpy().transpose(1,2,0) for i in input_list[:-1]]
			best['out'] = (preds[ind]>a).float().detach().cpu().squeeze().numpy().tolist()
			best['target'] = batch_c[ind].detach().cpu().numpy().squeeze()
			best['metrics'] = info['item_metrics'][-batch_len+ind]

		max_loss = torch.max(loss)
		if worst['loss'] is None or worst['loss'] > max_loss:
			ind = torch.argmax(loss)
			worst['loss'] = max_loss.item()
			worst['inp'] = [i[ind].detach().cpu().numpy().transpose(1,2,0) for i in input_list[:-1]]
			worst['out'] = (preds[ind]>a).float().detach().cpu().squeeze().numpy().tolist()
			worst['target'] = batch_c[ind].detach().cpu().numpy().squeeze()
			worst['metrics'] = info['item_metrics'][-batch_len+ind]
		
		if not k:
			ind = 0
			com['loss'] = loss[ind].item()
			#com['loss'] = loss.item()
			#print(com['loss'])
			com['inp'] = [i[0].detach().cpu().numpy().transpose(1,2,0) for i in input_list[:-1]]
			com['out'] = (preds[ind]>a).float().detach().cpu().squeeze().numpy().tolist()
			com['target'] = batch_c[ind].detach().cpu().numpy().squeeze()
			com['metrics'] = info['item_metrics'][-batch_len]
			k = True
		######################################
		"""

		samples_amount += batch_len
		info['loss'] += mean_loss.item()*batch_len
		info['item_loss'] += loss.detach().cpu().numpy().tolist()

	info['loss'] /= samples_amount
	if metrics != None:
		info['metrics'] = my_f1(info['table'])#/= samples_amount
		info['macro_metrics'] = f_macro(info['table'])
		info['micro_metrics'] = f_micro(info['table'])

	info['best'] = best
	info['worst'] = worst
	info['com'] = com
	return info


def visualizate(best, worst, com):
	h = len(best['inp'])
	w = 3
	names = ['hh', 'hv', 'autocor_hh', 'autocor_hv', 'autocor_hh_hv', 'autocor_hv_hh', 'target']
	fig = plt.figure(figsize=(w*5, h*5))
	k = 1
	for j in range(h):
		#print(j, k)
		ax = fig.add_subplot(h, w, k)
		if j != h-1:
			ax.imshow(best['inp'][j], cmap='gray')
		else:
			ax.imshow(best['inp'][j])
		ax.set_ylabel(names[j])
		if j == 0:
			ax.set_title('Loss: ' + str(best['loss'])[:4] + '\n' + 
						 'Metrics: ' + str(best['metrics'])[:4] + '\n' + 
						 'Classes: ' + str(best['target']) + '\n' +
						 'Preds: ' + str(best['out']))
		k += 3

	k = 2
	for j in range(h):
		ax = fig.add_subplot(h, w, k)
		if j != h-1:
			ax.imshow(worst['inp'][j], cmap='gray')
		else:
			ax.imshow(worst['inp'][j])
		if j == 0:
			ax.set_title('Loss: ' + str(worst['loss'])[:4] + '\n' + 
						 'Metrics: ' + str(worst['metrics'])[:4] + '\n' + 
						 'Classes: ' + str(worst['target']) + '\n' +
						 'Preds: ' + str(worst['out']))
		k += 3

	k = 3
	for j in range(h):
		ax = fig.add_subplot(h, w, k)
		if j != h-1:
			ax.imshow(com['inp'][j], cmap='gray')
		else:
			ax.imshow(com['inp'][j])
		if j == 0:
			ax.set_title('Loss: ' + str(com['loss'])[:4] + '\n' + 
						 'Metrics: ' + str(com['metrics'])[:4] + '\n' + 
						 'Classes: ' + str(com['target']) + '\n' +
						 'Preds: ' + str(com['out']))
		k += 3
	return fig

def train(epochs,
			model,
			opt,
			metrics,
			a,
			lr_scheduler,
			data_loaders,
			criterion,
			save_dir,
			device='cuda:0',
			):	
	
	last_epoch = epochs - 1
	#сохранение процесса обучения
	wdir = save_dir / 'checkpoints'
	wdir.mkdir(parents=True, exist_ok=True)
	last_path = wdir / 'last.pt'
	best_path = wdir / 'best.pt'
	training_process_file = save_dir / 'training_process.log'

	#тензорборд
	tb_logger = SummaryWriter(str(save_dir))
	tb_logger.add_hparams({'epochs':epochs, 'lr':get_lr(opt), 'a': a,
		 'train_bsize':data_loaders['train'].batch_size,}, {"hparams/f":1}, run_name = '.')

	last_lr = -1
	model.to(device)



	for epoch in range(epochs):
		last_epoch = (epoch == epochs - 1)
		print_message = []

		#Train step
		train_epoch_results = train_one_epoch(model, opt,data_loaders['train'],
												criterion, device, metrics, a)

		#Valid step
		valid_epoch_results = valid_one_epoch(model, data_loaders['valid'],
												criterion, device, metrics, a)
 
		if lr_scheduler is not None:
			lr_scheduler.step()
			last_lr = lr_scheduler.get_last_lr()[0]

		train_loss = train_epoch_results['loss']
		train_metrics = train_epoch_results['metrics']
		train_metrics_macro = train_epoch_results['macro_metrics']
		train_metrics_micro = train_epoch_results['micro_metrics']

		valid_loss = valid_epoch_results['loss']
		valid_metrics = valid_epoch_results['metrics']
		valid_metrics_macro = valid_epoch_results['macro_metrics']
		valid_metrics_micro = valid_epoch_results['micro_metrics']


		#Save
		if last_epoch:
			optimizer_class_name = {'optimizer_name': str(type(opt)).split('.')[-1].lower(),
									'base_optimizer_name': str(type(opt.base_optimizer)).split('.')[-1].lower()
														if hasattr(opt, 'base_optimizer') else None
									 }
			states = {
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': opt.state_dict(),
				'optimizer_class_name': optimizer_class_name,
				'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
			}
			if last_epoch:
				torch.save(states, last_path)
		

		valid_best = valid_epoch_results['best']
		valid_worst = valid_epoch_results['worst']
		valid_com = valid_epoch_results['com']

		#fig = visualizate(valid_best, valid_worst, valid_com)


		# fig = plt.figure(figsize=(12, 6))

		# ax_1 = fig.add_subplot(3, 3, 1)
		# ax_1.set(xticks=[], yticks=[], )
		# ax_1.xaxis.set_label_position('top')
		# ax_1.set_xlabel('Best')
		# ax_1.set_ylabel('Input')
		# ax_1.imshow(valid_best['inp'], vmin=0, vmax=255)

		# ax_2 = fig.add_subplot(3, 3, 2)
		# ax_2.set(xticks=[], yticks=[])
		# ax_2.xaxis.set_label_position('top')
		# ax_2.set_xlabel('Worst')
		# ax_2.imshow(valid_worst['inp'], vmin=0, vmax=255)

		# ax_3 = fig.add_subplot(3, 3, 3)
		# ax_3.set(xticks=[], yticks=[], )
		# ax_3.xaxis.set_label_position('top')
		# ax_3.set_xlabel('Com')
		# ax_3.imshow(valid_com['inp'], vmin=0, vmax=255)


		# ax_4 = fig.add_subplot(3, 3, 4)
		# ax_4.set(xticks=[], yticks=[], )
		# ax_4.xaxis.set_label_position('top')
		# ax_4.set_ylabel('Output')
		# ax_4.imshow(np.swapaxes(valid_best['out'], 2, 0)[:, :, : 3], vmin=0, vmax=255)

		# ax_5 = fig.add_subplot(3, 3, 5)
		# ax_5.set(xticks=[], yticks=[])
		# ax_5.xaxis.set_label_position('top')
		# ax_5.imshow(np.swapaxes(valid_worst['out'], 2, 0)[:, :, :4], vmin=0, vmax=255)

		# ax_6 = fig.add_subplot(3, 3, 6)
		# ax_6.set(xticks=[], yticks=[], )
		# ax_6.xaxis.set_label_position('top')
		# ax_6.imshow(np.swapaxes(valid_com['out'], 2, 0)[:, :, : 4], vmin=0, vmax=255)

		# ax_7 = fig.add_subplot(3, 3, 7)
		# ax_7.set(xticks=[], yticks=[], )
		# ax_7.xaxis.set_label_position('bottom')
		# ax_7.set_ylabel('Target')
		# ax_7.set_xlabel('Loss: ' + str(valid_best['loss']) + str(valid_best['out']))
		# ax_7.imshow(np.swapaxes(valid_best['target'], 2, 0)[:, :, : 4], vmin=0, vmax=255)

		# ax_8 = fig.add_subplot(3, 3, 8)
		# ax_8.set(xticks=[], yticks=[])
		# ax_8.xaxis.set_label_position('bottom')
		# ax_8.set_xlabel('Loss: ' + str(valid_worst['loss']) + str(valid_worst['out']))
		# ax_8.imshow(np.swapaxes(valid_worst['target'], 2, 0)[:, :, : 4], vmin=0, vmax=255)

		# ax_9 = fig.add_subplot(3, 3, 9)
		# ax_9.set(xticks=[], yticks=[], )
		# ax_9.xaxis.set_label_position('bottom')
		# ax_9.set_xlabel('Com')
		# ax_9.set_xlabel('Loss: ' + str(valid_com['loss']) + str(valid_com['out']))
		# ax_9.imshow(np.swapaxes(valid_com['target'], 2, 0)[:, :, : 4], vmin=0, vmax=255)

		"""
		tb_logger.add_figure('pred_examples/valid',
							 fig,
							 global_step=epoch,)
		"""
		#Print results to stdout and txt file
		print_message += [f'Epoch {epoch}\n']
		print_message += [f'Train loss: {round(train_loss, 7)}\n']
		print_message += [f'Train metrics: {round(train_metrics, 7)}\n']
		print_message += [f'Train metrics macro: {round(train_metrics_macro, 7)} \n']
		print_message += [f'Train metrics micro: {round(train_metrics_micro, 7)}\n']
		print_message += [f'Valid loss: {round(valid_loss, 7)}\n']
		print_message += [f'Valid metrics: {round(valid_metrics, 7)}\n']
		print_message += [f'Valid metrics macro: {round(valid_metrics_macro, 7)}\n']
		print_message += [f'Valid metrics micro: {round(valid_metrics_micro, 7)}\n']
		print_message += [f'Last lr: {last_lr}']
		with open(training_process_file, 'a+') as f:
			print(' | '.join(print_message), file=f)
			print(' | '.join(print_message), )
		
			print('\n', file=f)
			print('\n')
			

		#======= Output to tensorboard =========#


		tags = ['train/loss',  'valid/loss', 'train/metrics', 'train/metrics_macro', 'train/metrics_micro', 
		'valid/metrics', 'valid/metrics_macro', 'valid/metrics_micro']

		results = [train_loss, valid_loss, train_metrics, train_metrics_macro, train_metrics_micro, 
		valid_metrics, valid_metrics_macro, valid_metrics_micro]
		
		tags.append('lr')
		results.append(last_lr)
		
		for x, tag in zip(results, tags):
			tb_logger.add_scalar(tag, x, epoch)  

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

if __name__ == "__main__":
	#парсер входных аргументов
	args = parse_args()
	set_seed(args.seed)
	# Подключени видеокарты
	device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'
	if device != 'cpu':
		assert torch.cuda.is_available()

	#Данные
	data_loaders = dataset.get_dataloader(args)

	#Модель
	model = model.get_model()

	#Функция потерь
	if args.mse:
		criterion = nn.MSELoss(reduction='none')
	else:
		criterion = BCE #nn.CrossEntropyLoss(reduction='none')

	#Оптимизатор
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)


	experiment_dir = common.increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
	print('experiment_dir='+str(experiment_dir))

	# Save exp`s configuration to exp folder
	save_config_path = experiment_dir / 'config.yaml'
	with open(save_config_path, 'w+') as f:
		yaml.dump(vars(args), f, default_flow_style=False)

	metrics = None
	if args.f_score:
		print("dddddddddddddddddddddddd")
		metrics = f1_score

	train(epochs=args.epochs,
			model=model,
			opt=opt,
			metrics=metrics,
			a=args.a,
			lr_scheduler=None,
			data_loaders=data_loaders,
			criterion=criterion,
			save_dir=experiment_dir,
			device=device)
