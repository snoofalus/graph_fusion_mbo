
#builtin
import argparse
import os
from itertools import combinations

#torch
import torch
import torch.nn as nn
import torch.utils.data as data

#misc
import numpy as np
import matplotlib.pyplot as plt

#local
#import dataset.s1s2glcm as dataset
#import dataset.houston as dataset
import dataset.trento as dataset
from functions import *


parser = argparse.ArgumentParser(description='PyTorch Graph Fusion')

parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--iterations', default=500, type=int, metavar='N',
                    help='max possible number of iterations if MBO not converging')

parser.add_argument('--stopping-criteria', default=0.9999, type=float,
					help='percentage of change between iterations to stop above')

parser.add_argument('--error-margin', default=10**(-14), type=float, #0.1 def
					help='margin of error for lowest eigenval in nystrom')

#hyperparams
parser.add_argument('--n-landmark-nodes', default=50, type=int, metavar='N',#100 def
                    help='number of landmark nodes to approximate Laplacian')

parser.add_argument('--n-diffusions', default=3, type=int, metavar='N', # 0 or 5 def. current best 3
                    help='number of diffusion iterations between each threshold 1-10')

parser.add_argument('--mu', default=10**3, type=float, #10**2 or 10**3 def
					help='constant') 

parser.add_argument('--delta-t', default=0.1, type=float, #0.1 def
					help='timestep')

parser.add_argument('--semi-percent', default=0.1, type=float,#0.05-0.1 def
					help='percent of semi-supervised input')

args = parser.parse_args()

#True = nystrom laplacian
#False = laplacian
nystrom_bool = True
nystrom_drawing_methods = ['first', 'random', 'handpick']
nystrom_drawing_method = nystrom_drawing_methods[0]

#True = mbo
#False = spectral kmeans
mbo = True
mbo_drawing_methods = ['random', 'handpick'] #mbo needs all classes in semi, only use random if classes abundant
mbo_drawing_method = mbo_drawing_methods[1]


root = os.path.abspath('../data-local')

'''
ALGORITHM 1

Data: Co-registered data sets X1, X2...
Data: Number of desired classes m
Data: Semisupervised input u_hat
Result: Segmentation of X1,...,Xk into m classes

Calculate weighted graph representations W_1,..., W_k.

Fuse to one graph W representing the full input,
Section III-A1.

Apply Nyström method (Section III-D) to find graph Laplacian eigenvectors. 

Run Spectral Clustering (Section III-B) or Graph MBO (Section III-C)using eigenvectors.
'''

###
#S1S2
###-------------------------------------------------------------------------

#image1 = np.load(os.path.abspath('../data-local/images/s1s2seg/by-image/train+val/0.npy'))
#mask1 = np.load(os.path.abspath('../data-local/images/s1s2seg/by-image/train+valmask/0.npy'))


#C, H, W
#arr1 = image1[0:13, :, :] #s1
#arr2 = image1[13:, :, :] #s2


#print(s1.shape)
#print(s2.shape)

#print(image1.shape)
#print(mask1.shape)


###
#Houston
###-------------------------------------------------------------------------

'''
data_directory = os.path.join(root, 'images/houston/by-image/data')
mask_directory = os.path.join(root, 'images/houston/by-image/mask')


houston_dataset = dataset.get_houston(root)
loader = data.DataLoader(houston_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

#image1, mask1 = houston_dataset[390]#360


#create subplot with 6-9 plots to find good images


#groundtruth = np.load(os.path.join(root, 'workdir/houston/houston_groundtruth.npy')).T
#fig=plt.figure()
#plt.imshow(groundtruth)
#plt.show()
#exit()

norm = np.load(os.path.join(root,'workdir/houston/houston_norm.npy'))


image1 = np.load(os.path.abspath('data1.npy'))

channels, _, _ = image1.shape

for ch in range(channels):
	tmp = image1[ch, :, :] 
	image1[ch, :, :] = (tmp - norm[ch, 0])/norm[ch,1]


#add small constant for numerical stability
image1 = image1 + 0.1

mask1 = np.load(os.path.abspath('mask1.npy'))

arr1 = image1[:7, :, :]
arr2 = image1[7:, :, :]
'''

###
#Trento
###-------------------------------------------------------------------------

root = os.path.abspath('../data-local')

data_directory = os.path.join(root, 'images/trento/by-image/data')
mask_directory = os.path.join(root, 'images/trento/by-image/mask')


trento_dataset = dataset.get_trento(root) #len 48 if 64x64
loader = data.DataLoader(trento_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

#1, 7 
image1, mask1 = trento_dataset[7]#360


#create subplot with 6-9 plots to find good images

#fig=plt.figure()
#plt.imshow(mask1)
#plt.show()
#exit()

#add small constant for numerical stability
image1 = image1 + 0.1

arr1 = image1[:7, :, :]
arr2 = image1[7:, :, :]


###
#Random 
###-------------------------------------------------------------------------

#arr1 = np.random.rand(2, 16, 16)
#arr2 = np.random.rand(13, 16, 16)
#mask1 = np.random.randint(0, 6, (16, 16))

#arr1 = np.random.rand(7, 32, 32)
#arr2 = np.random.rand(144, 32, 32)
#mask1 = np.random.randint(0, 3, (32, 32))

#arr1 = np.random.rand(7, 64, 64)
#arr2 = np.random.rand(144, 64, 64)
#mask1 = np.random.randint(0, 5, (64, 64))

#120 worked
#arr1 = np.random.rand(7, 120, 120)
#arr2 = np.random.rand(144, 120, 120)
#mask1 = np.random.randint(0, 6, (120, 120))

###-------------------------------------------------------------------------

def main(mode1, mode2, mask, args):

	C1, H1, W1 = mode1.shape
	n_nodes = H1*W1

	new_mask_vector, mapping = maskToVector(mask)

	#each mode builds one graph of node distances 
	W1 = weightedGraph(mode1)
	W2 = weightedGraph(mode2)

	#fuse graphs by max and take exp
	W = fuseMatrix(W1, W2)

	if nystrom_bool == True:
		print('==> Calculating Nystrom Approximation\nn_landmark_nodes: [{}] drawn by: [{}] \n'.format(args.n_landmark_nodes, nystrom_drawing_method))

		if nystrom_drawing_method == 'first':

			#use first XX=args.n_landmark_nodes as landmark nodes
			W_idx = np.arange(n_nodes)
			XX_idx = W_idx[:args.n_landmark_nodes]
			YY_idx = W_idx[args.n_landmark_nodes:]

			#XX_idx = W_idx[-args.n_landmark_nodes:]
			#YY_idx = W_idx[:-args.n_landmark_nodes]

		elif nystrom_drawing_method == 'random':

			XX_idx, YY_idx = drawRandomNodes(n_nodes, args.n_landmark_nodes) 

		elif nystrom_drawing_method == 'handpick':

			XX_idx, YY_idx = drawHandpickedNodes(new_mask_vector, args.n_landmark_nodes)

		else:
			print("ERROR No valid drawing_method in Nystrom. Choose from ['first'], ['random'], ['handpick'].")
			exit()
		
		#approximate eigenvalues vector (I, ) and eigenvector matrix (I, K)
		tilde_eig, H = nystrom(W, XX_idx, YY_idx, args)

	else:
		print('==> Calculating True Laplacian')

		#calculate true L_sym = I - D**(-1/2)*W*D**(-1/2)
		tilde_eig, H = graphLaplacian(W)


	if mbo == True:

		n_mbo_draw = int(np.ceil(n_nodes*args.semi_percent))

		if mbo_drawing_method == 'random':

			mbo_idxs, _ = drawRandomNodes(n_nodes, n_mbo_draw) 
			mbo_labels = new_mask_vector[mbo_idxs]

		elif mbo_drawing_method == 'handpick':

			mbo_idxs, _ = drawHandpickedNodes(new_mask_vector, n_mbo_draw)
			mbo_labels = new_mask_vector[mbo_idxs]

		else:
			print("ERROR No valid drawing_method in MBO semisup. Choose from ['random'], ['handpick'].")
			exit()

		u_hat, Chi = createSemisupInput(mbo_idxs, mbo_labels, n_nodes)

		#returns assignment vector
		classification_vector = graphMBO(tilde_eig, H, u_hat, Chi, args)	
		classification_vector = mapBack(classification_vector, mapping)
		mask_hat = np.reshape(classification_vector, newshape=mask.shape)

		fig=plt.figure()
		plt.imshow(mask_hat)

		fig=plt.figure()
		plt.imshow(mask)
		plt.show()

	else:
		print("==> Running Spectral Clustering")
		#spectral kmeans on eigenvectors H, n_classes
		classification_vector = spectralClustering(H, len(mapping))
		classification_vector = mapBack(classification_vector, mapping)
		mask_hat = np.reshape(classification_vector, newshape=mask.shape)

		fig=plt.figure()
		plt.imshow(mask_hat)

		fig=plt.figure()
		plt.imshow(mask)
		plt.show()

main(arr1, arr2, mask1, args)




