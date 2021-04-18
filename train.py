#standard library
import os
import time
import argparse

#misc
import numpy as np
import matplotlib.pyplot as plt

#torch
import torch
import torch.nn as nn
import torch.utils.data as data

#local
from functions import *


parser = argparse.ArgumentParser(description='PyTorch Graph Fusion')

#run option hyperparams
parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--iterations', default=500, type=int, metavar='N',
                    help='max possible number of iterations if MBO not converging')

parser.add_argument('--stopping-criteria', default=0.9999, type=float,
					help='percentage limit of change between iterations to stop MBO')

parser.add_argument('--error-margin', default=10**(-14), type=float, 
					help='margin of error for lowest eigenval in nystrom')

parser.add_argument('--eigen_method', default='nystrom',
					help='Approximate eigenvectors with Nystrom or calculate True Laplacian: [nystrom], [laplacian].')

parser.add_argument('--nystrom_drawing_method', default='first',
					help='Method for drawing Nystrom landmark nodes: [first], [random], [handpick].')

parser.add_argument('--classifier', default='mbo',
					help='Classification method, i.e. graph mbo or spectral clustering: [spectral], [mbo].')

parser.add_argument('--mbo_drawing_method', default='handpick',
					help='Method for drawing semi-supervised input nodes: [random], [handpick].')

#model hyperparams
parser.add_argument('--n-landmark-nodes', default=50, type=int, metavar='N',#50, 100 defaults
                    help='number of landmark nodes to approximate Laplacian')

parser.add_argument('--n-diffusions', default=3, type=int, metavar='N', # 0 or 5 def. current best 3
                    help='number of diffusion iterations between each threshold 1-10')

parser.add_argument('--mu', default=10**3, type=float, #10**2 or 10**3 def
					help='constant representing confidence in semisupervised data') 

parser.add_argument('--delta-t', default=0.1, type=float, #0.1 def
					help='timestep for how fast diffusion occurs. Too small will freeze learning while too high fails to approximate Ginzburg-Landau')

parser.add_argument('--semi-percent', default=0.1, type=float,#0.05-0.1 def
					help='percent of semi-supervised input')

#datasets and pathing
parser.add_argument('--dataset', default='trento',
					help='Choose dataset from [trento], [houston], [s1s2].')

parser.add_argument('--result-directory', default='results',
					help='Choose dataset from [trento], [houston], [s1s2].')

args = parser.parse_args()


'''
ALGORITHM 1
###-------------------------------------------------------------------------
Data: Co-registered data sets X1, X2...
Data: Number of desired classes m
Data: Semisupervised input u_hat
Result: Segmentation of X1,...,Xk into m classes

Calculate weighted graph representations W_1,..., W_k.

Fuse to one graph W representing the full input,
Section III-A1.

Apply Nyström method (Section III-D) to find graph Laplacian eigenvectors. 

Run Spectral Clustering (Section III-B) or Graph MBO (Section III-C)using eigenvectors.
###-------------------------------------------------------------------------
'''

###
#Random inputs for testing
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

#120 largest currently working
#arr1 = np.random.rand(7, 120, 120)
#arr2 = np.random.rand(144, 120, 120)
#mask1 = np.random.randint(0, 6, (120, 120))
###-------------------------------------------------------------------------


def main():

	result_dir = os.path.abspath(os.path.join(args.result_directory, args.dataset))

	image_dir = os.path.join(result_dir, 'images')

	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	#load datasets: mbo does not train models so don't need separate train+test
	###-------------------------------------------------------------------------
	root = os.path.abspath('../data-local')

	if args.dataset == 'trento':
		print('==> Preparing trento segmentation dataset')

		import dataset.trento as dataset

		#possible display images 1, 7

		#num_channels = 77 #7 + 70
		modes1 = 7
		modes2 = 70
		num_classes = 7 #0-6

		dataset = dataset.get_trento(root)
		#loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 'houston':
		print('==> Preparing houston segmentation dataset')

		import dataset.houston as dataset

		#num_channels = 151 #7 + 144 
		modes1 = 7
		modes2 = 144
		num_classes = 16

		dataset= dataset.get_houston(root)
		#loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 's1s2':
		print('==> Preparing s1s1seg segmentation dataset')

		import dataset.s1s2 as dataset

		#num_channels = 15 #13+2
		modes1 = 13
		modes2 = 2
		num_classes = 6

		dataset = dataset.get_s1s2(root) #len 48 if 64x64
		#loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	else:
		print('Please pick a --dataset from [trento], [houston], [s1s2]')
		exit()


	if args.eigen_method == 'nystrom':
		print('==> Calculating Nystrom Approximation\nn_landmark_nodes: [{}] drawn by: [{}] \n'.format(args.n_landmark_nodes, args.nystrom_drawing_method))

	elif args.eigen_method == 'laplacian':
		print('==> Calculating True Laplacian')


	if args.classifier == 'mbo':
		print("==> Running Graph MBO \nsemisup: [{}] mu: [{}] dt: [{}] s: [{}]".format(args.semi_percent, args.mu, args.delta_t, args.n_diffusions))

	elif args.classifier == 'spectral':
		print("==> Running Spectral Clustering")

	#for display purposes
	#displayPredictionMaskPair(dataset, 41, modes1, num_classes, image_dir, args)

	#train and save to log file
	mean_acc, mean_iou = train(dataset, modes1, num_classes, result_dir, args)

def displayPredictionMaskPair(dataset, idx, modes1, num_classes, image_dir, args):

	image, mask = dataset[idx]

	#add small constant for numerical stability
	image = image + 0.1

	#split input into modes
	mode1 = image[:modes1]
	mode2 = image[modes1:]

	mask_hat = classify(mode1, mode2, mask, args)

	acc = pixelAccuracy(mask_hat, mask)
	class_iou, mean_iou = intersectionOverUnion(mask_hat, mask, num_classes)


	print('Acc: {:.3f}'.format(acc))

	print('IoU: {}'.format(class_iou))
	

	fig=plt.figure()
	plt.imshow(mask_hat)
	file_name = '{}_{}_mask.png'.format(args.dataset, idx)
	plt.savefig(os.path.join(image_dir, file_name))

	fig=plt.figure()
	plt.imshow(mask)
	file_name = '{}_{}_hat.png'.format(args.dataset, idx)
	plt.savefig(os.path.join(image_dir, file_name))

	plt.show()
	

def train(dataset, modes1, num_classes, result_dir, args):

	#init metrics
	num_images = 0.0
	all_accs = 0.0
	class_iou = []
	mean_iou = 0.0

	log_name = 'l{}_mu{}_dt{}_semi{}.txt'.format(args.n_landmark_nodes, args.mu, args.delta_t, args.semi_percent)
	log_path = os.path.join(result_dir, log_name)

	text_log = TextLog(log_path)
	text_log.headers(['Epoch', 'Total Ep', 'Current Acc', 'Mean Acc', 'Mean IoU' ])

	for epoch, (image, mask) in enumerate(dataset):

		#add small constant for numerical stability 
		image = image + 0.1

		#split multimode input into modes
		mode1 = image[:modes1]
		mode2 = image[modes1:]

		mask_hat = classify(mode1, mode2, mask, args)

		current_acc = pixelAccuracy(mask_hat, mask)
		all_accs = all_accs + current_acc
		num_images = num_images + 1
		mean_acc = all_accs / num_images


		class_iou, mean_iou = intersectionOverUnion(mask_hat, mask, num_classes)

		#takes in two lists: [epoch, total_epochs, current_accuracy, mean_accuracy, mean_iou], [class_iou]
		metrics = [epoch, len(dataset), current_acc, mean_acc, mean_iou]
		text_log.update(metrics, class_iou)

		if epoch % 4 == 0:
			print('Epoch: [{}/{}] mAcc: [{:.3f}] mIoU: [{:.3f}]'.format(epoch, len(dataset), mean_acc, mean_iou))
			print('class IoU{}'.format(class_iou))

	text_log.close()
	return mean_acc, mean_iou

def classify(mode1, mode2, mask, args):

	C1, H1, W1 = mode1.shape
	n_nodes = H1*W1

	new_mask_vector, mapping = maskToVector(mask)

	#each mode builds one graph of node distances 
	W1 = weightedGraph(mode1)
	W2 = weightedGraph(mode2)

	#fuse graphs by max and take exp
	W = fuseMatrix(W1, W2)

	if args.eigen_method == 'nystrom':
		#print('==> Calculating Nystrom Approximation\nn_landmark_nodes: [{}] drawn by: [{}] \n'.format(args.n_landmark_nodes, args.nystrom_drawing_method))

		if args.nystrom_drawing_method == 'first':

			#use first XX=args.n_landmark_nodes as landmark nodes
			W_idx = np.arange(n_nodes)
			XX_idx = W_idx[:args.n_landmark_nodes]
			YY_idx = W_idx[args.n_landmark_nodes:]

			#XX_idx = W_idx[-args.n_landmark_nodes:]
			#YY_idx = W_idx[:-args.n_landmark_nodes]

		elif args.nystrom_drawing_method == 'random':

			XX_idx, YY_idx = drawRandomNodes(n_nodes, args.n_landmark_nodes) 

		elif args.nystrom_drawing_method == 'handpick':

			XX_idx, YY_idx = drawHandpickedNodes(new_mask_vector, args.n_landmark_nodes)

		else:
			print("ERROR No valid drawing_method in Nystrom. Choose from ['first'], ['random'], ['handpick'].")
			exit()
		
		#approximate eigenvalues vector (I, ) and eigenvector matrix (I, K)
		tilde_eig, H = nystrom(W, XX_idx, YY_idx, args)

	elif args.eigen_method == 'laplacian':
		#print('==> Calculating True Laplacian')

		#calculate true L_sym = I - D**(-1/2)*W*D**(-1/2)
		tilde_eig, H = graphLaplacian(W)


	if args.classifier == 'mbo':

		n_mbo_draw = int(np.ceil(n_nodes*args.semi_percent))

		if args.mbo_drawing_method == 'random':

			mbo_idxs, _ = drawRandomNodes(n_nodes, n_mbo_draw) 
			mbo_labels = new_mask_vector[mbo_idxs]

		elif args.mbo_drawing_method == 'handpick':

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

		return mask_hat

	elif args.classifier == 'spectral':

		#print("==> Running Spectral Clustering")
		#spectral kmeans on eigenvectors H, n_classes
		classification_vector = spectralClustering(H, len(mapping))
		classification_vector = mapBack(classification_vector, mapping)
		mask_hat = np.reshape(classification_vector, newshape=mask.shape)

		return mask_hat

	else:
		print('ERROR No valid classifier, choose from [spectral], [mbo]')


main()




