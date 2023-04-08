import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#from tkinter import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', type=str, help='path with files of training')
	parser.add_argument('-t', '--type-of-graph', type=str, help='type of graph for converting to csv (valid/loss, train/metrics)')
	parser.add_argument('--name', default='temp', type=str, help='name of csv file')
	args = parser.parse_args()
	cfg = vars(args)
	args = argparse.Namespace(**cfg)
	print('\n', args, '\n')
	
	assert args.data is not None
	assert args.type_of_graph is not None
	return args


if __name__ == "__main__":
    args = parse_args()
    log_dir = args.data
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    events = event_accumulator.Scalars(args.type_of_graph)
    x = [x.step for x in events]
    y = [x.value for x in events]
    df = pd.DataFrame({"step": x, "log_dir": y})

    df.to_csv(args.name + ".csv")
    
    plt.plot(x, y)
    plt.show()
