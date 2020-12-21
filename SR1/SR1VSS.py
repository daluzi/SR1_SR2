# _*_ coding: utf-8 _*_
# @Author   : daluzi
# @time     : 2019/10/25 21:29
# @File     : SR1VSS.py
# @Software : PyCharm

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import *
import copy
import metrices
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
import random


# 相似矩阵
def trainW(v):
	similarMatrix = cosine_similarity(v)
	m = np.shape(similarMatrix)[0]
	for i in range(m):
		for j in range(m):
			if j == i:
				similarMatrix[i][j] = 0
	return similarMatrix

# KNN
def myKNN(S, k):
	N = len(S)  # 输出的是矩阵的行数
	A = np.zeros((N, N))

	for i in range(N):
		dist_with_index = zip(S[i], range(N))
		dist_with_index = sorted(dist_with_index, key=lambda x: x[0], reverse=True)
		# print(dist_with_index)
		neighbours_id = [dist_with_index[m][1] for m in range(k)]  # xi's k nearest neighbours
		# print("neigh",neighbours_id)
		for j in neighbours_id:  # xj is xi's neighbour
			# print(j)
			A[i][j] = 1
			# A[j][i] = A[i][j]  # mutually
	# print(A[i])
	m = np.shape(A)[0]
	for i in range(m):
		for j in range(m):
			if j == i:
				A[i][j] = 0
	return A

def Update(R, k, r, lamb1, lamb2, aerfa, aerfa1):
	'''
	:param R: user-item matrix
	:param k: The number of iterations
	:param r: r-rank factors
	:param lamb1: used to calculate U
	:param lamb2: used to calculate V
	:param aerfa:
	:return:
	'''
	print("R:\n", R)
	I = copy.copy(R)
	I[I > 0] = 1

	m, n = R.shape
	U = np.array(np.random.random((r, m)))
	V = np.array(np.random.random((r, n)))

	# 这里可以通过KNN找到user的朋友矩阵
	simiX = trainW(R)
	W = myKNN(simiX, 5)
	# print("w:",W)

	# updating formulas
	for i in range(k):
		# U
		for i_u in range(m):
			subU1 = np.zeros((r, 1))
			for j_u in range(n):
				# print(np.array(U[:,i_u].T).shape, np.array(V[:,j_u]).shape)
				# print(U[:,j_u].T)
				# print(V[:,j_u])
				# print(I[i_u][j_u])
				subU1 = subU1 + (I[i_u][j_u] * (np.dot(U[:, i_u].T, V[:, j_u]) - R[i_u][j_u])) * V[:, j_u]
			subU1 = subU1 + lamb1 * U[:, i_u]

			subU2on = np.zeros((r, 1))
			subU2down = 0
			Fri = np.argwhere(W[i_u] == 1)
			# print(len(Fri))
			for f in range(len(Fri)):
				# print(Fri[f][0])
				# print("i_u\t",i_u)
				# print("Fri[%d][0]\t" % f,Fri[f][0])
				# print(U[:,Fri[f][0]].shape)
				# print("asdasdd:\n",subU2on)
				subU2on = subU2on + (metrices.VectorSpaceSimilarity(R, I, i_u, Fri[f][0])) * np.array(U[:, Fri[f][0]])
				# print("asd",U[:,Fri[f][0]])
				# print("")
				subU2down = subU2down + metrices.VectorSpaceSimilarity(R, I, i_u, Fri[f][0])
			# print("subU2on:\n", subU2on)
			# print("subU2down:\n", subU2down)
			subU2 = aerfa * np.array(U[:, i_u] - (subU2on[0] / subU2down))

			subU3 = np.zeros((r, 1))
			subU3on = np.zeros((r, 1))
			subU3down = 0
			subU3onon = np.zeros((r, 1))
			subU3downdown = 0
			for g in range(len(Fri)):
				# for f in range(len(Fri)):
				# subU3onon = subU3onon + (metrices.VectorSpaceSimilarity(R, I, Fri[g][0], Fri[f][0])) * U[:,Fri[f][0]]
				# subU3downdown = subU3downdown + metrices.VectorSpaceSimilarity(R, I, Fri[g][0], Fri[f][0])
				# subU3down = subU3down + metrices.VectorSpaceSimilarity(R, I, Fri[g][0], Fri[f][0])
				subU3on = subU3on + -(metrices.VectorSpaceSimilarity(R, I, i_u, Fri[g][0]) * np.array(
					U[:, g] - (subU2on[0] / subU2down)))
				subU3 = subU3 + (subU3on / subU2down)
			# print("subU3 run\t\t:",g)
			subU3 = aerfa * np.array(subU3)

			subU = subU1 + subU2 + subU3

			# print("subU1:\n",subU1)
			# print("subU2:\n",subU2)
			# print("subU3:\n",subU3)
			# print("subU:\n",subU)
			U[:, i_u] = U[:, i_u] - aerfa1 * np.array(subU[0])

		# V
		for i_v in range(n):
			subV = np.zeros((1, r))
			for j_v in range(m):
				subV = subV + (I[j_v][i_v] * (np.dot(U[:, j_v].T, V[:, i_v]) - R[j_v][i_v])) * U[:, j_v]
			subV = subV + lamb2 * V[:, i_v]
			# print("subV:\n", subV)
			V[:, i_v] = V[:, i_v] - aerfa1 * subV[0]
		print("run%d" % i)

	return U, V

