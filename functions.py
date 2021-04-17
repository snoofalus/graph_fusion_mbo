
#misc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

def maskToVector(mask):
	'''
	Takes in groundtruth mask of an image and returns it as vector.
	If the classes are not ordered with missing classes between those present, e.g. if classes in image are 0,1,3
	the resulting mask vector will map those to 0,1,2 since MBO degrades for increased number of classes.

	Args:
		mask: groundtruth image

	Output:
		new_mask_vector: flattened and mapped mask 
		mapping: dictionary where each key is the original lable and 
		their corresponding value is that labels new mapping
	'''

	old_mask_vector = np.ndarray.flatten(mask)

	unique = np.unique(old_mask_vector)
	n_unique = len(unique)
	range_array = np.arange(n_unique)

	mapping = {}

	#create dictionary mapping, i.e. 1:2
	for key, value in zip(unique, range_array):
		mapping.update({key:value})

	if np.array_equal(unique, range_array):
		#print('Mapping  not needed')
		return old_mask_vector, mapping


	#alternative to unique mapping approach below
	#new_mask_vector = np.vectorize(mapping.get)(old_mask_vector)


	#this mapping is less intuitive but extremely fast when (n_unique << elements in array)
	u, inv = np.unique(old_mask_vector, return_inverse = True)
	new_mask_vector = np.array([mapping[x] for x in u])[inv].reshape(old_mask_vector.shape)

	return new_mask_vector, mapping

def distance(vi, vj):
	#euclidean distance between vertices is l2 norm
	dist = np.linalg.norm(vi - vj)
	return dist

def drawRandomNodes(n_nodes, n_draw):
	idxs = np.arange(n_nodes) 
	np.random.shuffle(idxs)

	XX_idx = np.sort(idxs[:n_draw])
	YY_idx = np.sort(idxs[n_draw:])

	return XX_idx, YY_idx

def drawHandpickedNodes(mask_vector, n_draw):
	'''
	Draw n_landmark_nodes as balanced as possible
	from the m classes in mask_vector.

	Output:
		XX_idx, YY_idx: sorted nodes for Nystrom where XX are landmark
		indices and YY are the remaining nodes
	'''

	unique = np.unique(mask_vector)
	XX_idx = []

	#create list of arrays where array i has all indices of class i
	list_of_arrays = [np.where(mask_vector == label)[0] for label in unique]

	n_classes_left = len(unique)
	n_left_to_pick = n_draw

	while n_left_to_pick > 0:

		#if only one class left draw n_left_to_pick random indices from that class
		if len(list_of_arrays) == 1:

			array = list_of_arrays[0]

			np.random.shuffle(array)

			if n_left_to_pick == 1:
				XX_idx.append(array[:n_left_to_pick][0])

			else:
				XX_idx.extend(array[:n_left_to_pick])

			n_left_to_pick = n_left_to_pick - n_left_to_pick 
			break

		#else draw (n=len smallest class) balanced from all classes left
		else:

			n_pick_per_class = n_left_to_pick // n_classes_left

			#special case for n_left_to_pick < n_smallest_classes left leads to 0 picked
			if n_pick_per_class == 0:
				n_pick_per_class = 1
			
			#init as the largest possible val
			n_smallest_class = len(mask_vector)

			#find n_smallest class
			for array in list_of_arrays:

				if len(array) < n_smallest_class:
					n_smallest_class = len(array)

			if n_smallest_class < n_pick_per_class:
				n_pick_per_class = n_smallest_class

			#empty arrays at deletion indeces are deleted after draw
			del_indeces = []

			#draw n_pick_per_class indxs from all non empty classes, add to XX, remove those indxs
			for i, array in enumerate(list_of_arrays):

				#draw n_pick_per_class randomly from current array
				np.random.shuffle(array)#might be better to do this at creation of arrays

				if n_pick_per_class == 1:
					XX_idx.append(array[:n_pick_per_class][0])
					#print(array[:n_pick_per_class][0])

				else:
					XX_idx.extend(array[:n_pick_per_class])
					#print(array[:n_pick_per_class])

				list_of_arrays[i] = array[n_pick_per_class:]

				n_left_to_pick = n_left_to_pick - n_pick_per_class

				if len(list_of_arrays[i]) == 0:
					del_indeces.append(i)
					n_classes_left = n_classes_left - 1

				if n_left_to_pick == 0:
					break

			if len(del_indeces) > 0:
				#remove empty arrays from list
				for del_idx in sorted(del_indeces, reverse=True):
					del list_of_arrays[del_idx]

	XX_idx = np.sort(XX_idx)

	#YY is the indices from total indices not in XX
	YY_idx = np.arange(len(mask_vector))
	YY_idx = np.setdiff1d(YY_idx, XX_idx)

	return XX_idx, YY_idx

def weightedGraph(data_modality):
	#reshape into x datapoints with y dimensions
	C, H, W = data_modality.shape

	#total nodes
	n = H*W

	#pdist takes data in (n points, m dim space)
	data_flatten = np.reshape(data_modality, (C, n))
	data = np.transpose(data_flatten)

	#distance vector elements = (n-1) + (n-2)+...
	v = pdist(data, 'euclidean')

	#similarity matrix W
	W = squareform(v)

	#scale sets to make distances comparable
	W = scale(W)

	return W

def scale(W):

	scaling = np.std(W)

	W = W/scaling

	return W

def fuseMatrix(W1, W2):

	W = np.maximum(W1, W2)

	W = np.exp(-W)

	return W

def nystrom(W, XX_idx, YY_idx, args, flip=False):
	'''
	Generally follows notation of page 308 in:
	https://www.researchgate.net/publication/301941366_Diffuse_Interface_Models_on_Graphs_for_Classification_of_High_Dimensional_Data

	Approximates graph laplacian eigenvectors from 
	the weighted graph W(NxN) with L landmark nodes of fused multimodal input X_1, ..., X_k.

	This is done by projecting the majority of calculations 
	lying in W_YY from landmark nodes s.t. with L landmark nodes the largest marix calculations are
	at most L by N and not N by N.

	L is either drawn at random, found by k-means or specifically chosen 
	to represent most of the classes of an image.

	Args:
		W: total fused weight matrix
		XX_idx: L by 1 vector with names of sampled landmark nodes from W
		YY_idx: remaining nodes from W
		algo: svd or eigh

	Output:
		Sigma: vector of L eigenvalues
		Phi: N by L eigenvector matrix with i'th column being an eigenvec belonging to i'th eigenvalue
	'''

	#permutation vectors of indices between X->X and X->Y
	XX_perm = np.array(np.meshgrid(XX_idx, XX_idx)).T.reshape(-1, 2)
	XY_perm = np.array(np.meshgrid(XX_idx, YY_idx)).T.reshape(-1, 2)

	#N total nodes 
	N = W.shape[0] #n_W

	#L interpolation points/landmark nodes
	L = len(XX_idx) #n_XX

	#remaining N-L nodes
	NminL = N - L #n_YY

	W_XX = np.zeros((L, L))
	c = 0
	for i in range(L):
		for j in range(L):
			W_XX[i, j] = W[XX_perm[c][0], XX_perm[c][1]]
			#print("{}, {}".format(XX_perm[c][0], XX_perm[c][1]))
			c = c + 1

	W_XY = np.zeros((L, NminL))
	c = 0
	for i in range(L):
		for j in range(NminL):
			W_XY[i, j] = W[XY_perm[c][0], XY_perm[c][1]]
			c = c + 1

	#if needed can add small ridge to W_XX here for numerical stability

	#unit vectors
	v_x = np.ones((L, 1))
	#v_x = v_x / np.linalg.norm(v_x)

	v_y = np.ones((NminL, 1))
	#v_y = v_y / np.linalg.norm(v_y)
	

	#d_X = W_XX*1_L + W_XY*1_(N-L)
	d_X = np.matmul(W_XX, v_x) + np.matmul(W_XY, v_y)

	#d_Y = W_YX*1_L + (W_YX*W_XX^(-1)*W_XY)1_(N-L)
	d_Y = np.matmul(W_XY.T, v_x) + np.linalg.multi_dot([W_XY.T, np.linalg.inv(W_XX), W_XY, v_y])

	s_X = np.sqrt(d_X)
	s_Y = np.sqrt(d_Y)


	#normalized matrices 

	#W_hatXX = W_XX./(s_X*s_X^T)
	W_hatXX = np.divide(W_XX, np.matmul(s_X, s_X.T))

	#W_hatXY = W_XY./(s_X*s_Y^T)
	W_hatXY = np.divide(W_XY, np.matmul(s_X, s_Y.T))

#flip eigvals
###-------------------------------------------------------------------------

	#W_hatXX = B_X*Gamma*B_X^T

	B_X, Gamma_eig, B_X_T = np.linalg.svd(W_hatXX)

	if flip == True:

		#reverse eigenvals to ascending
		Gamma_eig = np.flip(Gamma_eig)

		#flip columns
		B_X = np.flip(B_X, axis=1)

		#flip rows
		B_X_T = np.flip(B_X_T, axis=0)

###-------------------------------------------------------------------------

	#prepare Gamma^(-1/2), Gamma^(1/2))
	Gamma_neg = np.diag(1 / (np.sqrt(Gamma_eig)))
	Gamma_pos = np.diag(np.sqrt(Gamma_eig))

	#S = B_X*Gamma^(-1/2)*B_X^T
	S = np.linalg.multi_dot([B_X, Gamma_neg, B_X_T])

	#Q = W_hatXX + S*(W_hatXY*W_hatXY^T)*S
	Q1 = np.matmul(W_hatXY, W_hatXY.T)
	Q = W_hatXX + np.linalg.multi_dot([S, Q1, S])

#flip eigvals
###-------------------------------------------------------------------------

	#Q = A*Xi*A^T

	A, Xi_eig, _ = np.linalg.svd(Q)

	if flip == True:

		#reverse eigenvals to ascending
		Xi_eig = np.flip(Xi_eig)

		#flip columns
		A = np.flip(A, axis=1)

###-------------------------------------------------------------------------

	#prepare Xi^(-1/2)
	Xi_neg = np.diag(1 / np.sqrt(Xi_eig))

	#Phi_num = B_X*Gamma^(1/2)*B_X^T*(A*Xi^(-1/2))
	Phi_numerator = np.linalg.multi_dot([B_X, Gamma_pos, B_X_T, A, Xi_neg])

	#Phi_den = W_hatXY.T*B_X*Gamma^(-1/2)*B_X^T*(A*Xi^(-1/2))
	Phi_denom = np.linalg.multi_dot([W_hatXY.T, B_X, Gamma_neg, B_X_T, A, Xi_neg])

	Phi = np.concatenate((Phi_numerator, Phi_denom), axis=0)

	tilde_eig = 1 - Xi_eig 

	#print(tilde_eig[0])
	#print(tilde_eig[1])
	#print(tilde_eig[-2])
	#print(tilde_eig[-1])
	#exit()

	if flip == True:
		#reverse eigenvals to ascending
		tilde_eig = np.flip(tilde_eig)

		#flip columns
		Phi = np.flip(Phi, axis=1)


	if (tilde_eig[0] < 0) and (abs(tilde_eig[0]) < args.error_margin):
		#print("Negative eigval [{}] in Nystrom within margin of error.".format(tilde_eig[0]))
		tilde_eig[0] = 0

	elif (tilde_eig[0] < 0) and (abs(tilde_eig[0]) > args.error_margin):
		print("ERROR Negative eigval [{}] in Nystrom. Please try: [adding small value to image], [adding small ridge to diag of Wxx] or [Different normalization of image].".format(tilde_eig[0]))
		exit()
	
	return tilde_eig, Phi
	

def graphLaplacian(W):
	'''
	Function for comparing Nystrom approximation to true values of 
	symmetric normalized graph laplacian.
	The symmetric graph Laplacian is def:
	L_sym = I - D^(-1/2)*W*D^(-1/2)

	Args:
		W: weighted graph similarity matrix

	Outputs:
		eigenvalues, eigenvectors of true symmetric graph Laplacian
	'''
	
	#d_i = sum_j w_ij
	d = np.sum(W, axis = 0)

	#included in a similar algo for numerical stability in normalized sym laplacian but doesnt seem to have effect
	#d += np.spacing(np.array(0, W.dtype))

	D_neg = np.diag(1/(np.sqrt(d)))
	I = np.eye(W.shape[0])

	#L_sym = I - D^(-1/2)*W*D^(-1/2)
	L_sym = I - np.linalg.multi_dot([D_neg, W, D_neg])

	#SVD = u, s, vt
	eigvec, eigval, _ = np.linalg.svd(L_sym)

	#EigDecomp = Bx*W*Bx^T
	#eigval, eigvec = np.linalg.eigh(L_sym)


	#reverse eigenvals to ascending order
	eigval = np.flip(eigval)

	#flip columns accordingly
	eigvec = np.flip(eigvec, axis=1)

	#print(eigval[0])
	#print(eigval[1])
	#print(eigval[-2])
	#print(eigval[-1])
	#exit()
	
	return eigval, eigvec

def check_symmetric(a, rtol=1e-05, atol=1e-08):
	#check if matrix a is symmetric
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_pos_def(x):
	#check if if matrix positive definite
    return np.all(np.linalg.eigvals(x) > 0)

def spectralClustering(H, n_centroids):
	'''
	KMeans spectral clustering, mainly included as source of comparison
	'''

	#H is (n_samples, n_dims)

	kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(H)
	classification_vector = kmeans.labels_

	return classification_vector

def createSemisupInput(idxs, labels, n_nodes):
	'''
	Creates semisupervised input u_hat for MBO algorithm from groundtruth pixels.
	Generalized to not take in mask but only 5-10% pairs of (pixel, class) semi-input.

	Args:
		idxs: vector of indeces for groundtruth to include as semisup input where each pixel is an int in [1,...,m]. 
		labels: groundtruth labels corresponding to idxs.
		n_nodes: total number of nodes.
	Output:
		u_hat:  (n, m) matrix of 0s and 1s where each row belongs to a pixel 
		and has a single 1 per row to indicate class
		Chi: (n, ) vector similar to u_hat which tells if point i belongs to fidelity or not.
		Can alternatively be extracted from u_hat.
	'''
	
	n_classes = len(np.unique(labels))

	u_hat = np.zeros((n_nodes, n_classes))
	Chi = np.zeros((n_nodes,))

	#change only fidelity rows
	for idx, element in zip(idxs, labels):
		u_hat[idx, element] = 1
		Chi[idx] = 1

	return u_hat, Chi

def graphMBO(tilde_eig, H, u_hat, Chi, args):
	'''
	Segmentation of i total nodes of W into m classes.
	Done by classifying k eigenvalues(tilde_eig) and -vectors(H matrix) approximated by Nyström
	along with semisupervised input u_hat ~1-10% of nodes sampled from V.

	Args:
		tilde_eig: vector of L ascending eigenvectors 
		H (Phi): i by k eigenvectors, k'th column = k'th eigenvec
		u_hat: (i, m) matrix of semisupervised input


	M=number of expected classes
	I=number of nodes
	K=number of approximated eigenvalues and vectors from nyström
	'''

	M = u_hat.shape[1] 
	I = H.shape[0]
	K = H.shape[1]

	#print("==> Running Graph MBO on {} nodes with {} classes\nsemisup: [{}] mu: [{}] dt: [{}] s: [{}]".format(I, M, args.semi_percent, args.mu, args.delta_t, args.n_diffusions))

	u = np.zeros((u_hat.shape))
	current_labels = np.zeros((I,))

	
	# method 1: init u^0 entirely random separate from u_hat
	for i in range(I):
		idx = np.random.randint(M)
		u[i, idx] = 1
		current_labels[i] = idx
	
	'''
	# method 2: init u_hat = u^0, random except fidelity rows
	for i in range(I):

		#part of fidelity
		if np.any(u_hat[i, :])==True:
			idx = np.random.randint(2)
			u[i, idx] = 1

		#not part of fidelity
		else:
			u[i, :] = u_hat[i, :]
	u_hat = u
	'''
	
	#initialize d^0, (k, m) matrix
	d = np.zeros((K, M))

	#denom does not change between iterations, calculate outside
	#denom[k] = 1 + mu*dt + dt*lambda[k]
	denom = 1 + args.mu*(args.delta_t/args.n_diffusions) + (args.delta_t/args.n_diffusions)*tilde_eig

	for iteration in range(args.iterations):

		#if iteration % 10 == 0:
			#print("\nEpoch: [{}|{}]".format(iteration, args.iterations))

		# a = H^T*u
		a = np.matmul(H.T, u)

		#diffusion step, diffuse once or s times
		for s in range(args.n_diffusions):

			nom = (1 + args.mu*(args.delta_t/args.n_diffusions))*a - args.mu*(args.delta_t/args.n_diffusions)*d

			#use data / vector[:,None] to divide each row in matrix data by corresponding element of vector 
			a = nom/denom[:, None]
			#a = (nom.T / denom).T

			#u^{n+1/2} = H*a
			u = np.matmul(H, a)

			#tmp = (u-u_hat)*Chi[:, None]
			tmp = ((u-u_hat).T*Chi).T
			#d^n = H^T Chi(x)(u^n-u_hat)
			d = np.matmul(H.T, tmp)
		

		#thresholding step
		u, current_labels, percent_equal = threshold(u, current_labels)

		#stop and return u if difference between iterations < puritymeasure
		if percent_equal > 0.9999:
			#print("Graph MBO converged at iter: {}".format(iteration))
			break

	return current_labels 

def threshold(u, current_labels):
	'''
	Args:
		u: Current halfstep assignment matrix (I, M)of probabilties of each class. 
		Elements of u halfstep can take on real values as opposed to threshold steps.

		current_labels: Vector (I, ) where each element is a class number
		representing the node belonging from the last performed thresholding.
	Out:
		u_next: Thresholded matrix where each row is the standard basis vector
		e_r according to max probability class r, s.t. u_i^{n + 1} = e_r, 
		r = argmax_j u_ij^(n + 1/2)

		next_labels: Same as current labels for next iteration.

		stoppingCriteria: Not a returned value, checks purtiy between current and next 
		labels and stops diffuse/thresholding if less than 99.99% change between iters
	'''

	u_next = np.zeros((u.shape))
	next_labels = np.zeros((len(current_labels), ))

	#probabilities per row should sum to 1
	#print(u[0,:])
	#exit()

	#next prediction from thresholding
	for i in range(u.shape[0]):
		idx = np.where(u[i, :] == np.amax(u[i, :]))[0][0]
		#print(np.where(u[i, :] == np.amax(u[i, :])))
		#exit()

		u_next[i, idx] = 1
		next_labels[i] = idx

	#print(u_next[30,:])

	percent_equal = stoppingCriteria(current_labels, next_labels)

	return u_next, next_labels, percent_equal

def stoppingCriteria(current_labels, next_labels):

	n_equal = np.sum(current_labels == next_labels)
	percent_equal = n_equal / len(current_labels)
	#print(percent_equal)

	return percent_equal

def mapBack(new_mask_vector, mapping):

	inv_map = {v: k for k, v in mapping.items()}


	#alternative to unique mapping approach below
	#old_mask_vector = np.vectorize(inv_map.get)(new_mask_vector)


	#this mapping is less intuitive but extremely fast when (n_unique << elements in array)
	u, inv = np.unique(new_mask_vector, return_inverse = True)
	old_mask_vector = np.array([inv_map[x] for x in u])[inv].reshape(new_mask_vector.shape)

	return old_mask_vector

def pixelAccuracy(mask_hat, mask):
	'''
	Calculates pixel accuracy 

	Args: 
		logits: (N, classes, H, W)
		masks: (N, H, W)
	'''

	correct_pixels = np.sum(np.equal(mask_hat, mask))
	total_pixels = np.prod(mask.shape)

	accuracy = correct_pixels / total_pixels

	return accuracy

def intersectionOverUnion(mask_hat, mask, num_classes):
	'''
	Takes in vector of groundtruth and vector of predicted classes and uses
	those to calculate Intersection over Union and mean IOU.

	true_positive = mask vector and classification agrees
	false_positive = predicted class i but belongs to class !=i
	false_negative = belongs to class i but predicted class !=i

	IOU by tp / (tp + fp + fn)

	mean IOU is found by summing the IOU over all classes and dividing by 
	number of classes.

	Args:
		mask_vector: vector of groundtruth labels
		classification_vector: vector of model predicted labels
	Output:
		iou: vector of intersection over union values describing how well 
		each prediction performance per class (except background which is ignored as 
		it can contain other classes for many datasets).
		mean_iou: number describing overall prediction performance
	'''

	class_iou = []#np.empty((num_classes, ))

	#start from 1 to ignore background = 0
	for cl in range(num_classes):

		true_positive = np.sum(np.logical_and(mask_hat == cl, mask == cl))
		false_positive = np.sum(np.logical_and(mask_hat != cl, mask == cl))
		false_negative = np.sum(np.logical_and(mask_hat == cl, mask != cl))
		ground_truth = true_positive + false_positive + false_negative

		if ground_truth != 0:
			class_iou.append(true_positive / ground_truth)

		else:
			class_iou.append('nan')

		#if (true_positive == 0) & (false_positive == 0) & (false_negative == 0): # possibly iou_class = 1
		#  tmp_class_iou = 0

	mean_iou = 0.0
	#mean_iou = np.sum(class_iou) / (n_unique - 1)

	return class_iou, mean_iou

class TextLog(object):
	def __init__(self, log_path):
		#remember to use .close method after initializing object
		self.file = open(log_path, 'w')

	def headers(self, headers):
		#creates headliners for log file
		#self.metrics = {}

		for head in headers:
			self.file.write(head)
			self.file.write('\t')
			#self.metrics[head] = []
		self.file.write('\n')
		self.file.flush()

	def update(self, metrics, class_iou):
		#save (epoch, total_epochs, accuracy, mean_iou) and class ious on alternating lines.
		for metric in metrics:
			self.file.write('{}'.format(metric))
			self.file.write('\t')
		self.file.write('\n')

		for class_idx, iou in enumerate(class_iou):
			if iou == 'nan':
				self.file.write('{}: nan'.format(class_idx))
			else:
				self.file.write('{}: {:.3f}'.format(class_idx, iou))
			self.file.write('\t')

		self.file.write('\n')

		self.file.flush()

	def close(self):
		self.file.close()
