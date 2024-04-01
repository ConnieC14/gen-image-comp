import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--count', type=int, default=1, help='Number of results to evaluate on')
args = parser.parse_args()


data_list = np.array([i for i in range(1,args.count+1)])
print("Created ID list with length: ", len(data_list))
np.save('test_bench/id_list.npy', data_list)
