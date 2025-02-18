from .HMM import HMM

import autograd.numpy as np
from autograd import elementwise_grad as egrad
from matplotlib import pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_rank as rank
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace
from autograd.numpy import pi
from matplotlib.patches import Circle
import os
import sys


class Kraus_HMM(HMM):
    def __init__(self,
                 nql,
                 num_observations,
                 random_seed = 12345,
                 ):
        self.nql = nql
        self.num_observations = num_observations
        self.random_seed = random_seed
        
    @staticmethod
    def Linear_Dependent(M):
        """
        判断一个矩阵中的行向量是否线性相关
        :param M: 矩阵
        :return: 布尔值，True:线性相关， False:线性无关
        """
        if rank(M) < min(M.shape):
            return True
        else:
            return False

    def Generate_Kraus(self):
        """
        产生初始的的Kraus算符
        :return:
        """

        np.random.seed(self.random_seed)

        A = np.random.random([self.nql * self.num_observations, self.nql * 2])

        ref = self.Linear_Dependent(A)
        B = np.transpose(A)
        if ref:
            print("Linear Dependent!")
            quit()
        else:
            # 对每一行进行施密特正交化,第一行为基础行，不进行正交
            for i in range(1, B.shape[0]):
                for j in range(0, i):
                    B[i, :] = B[i, :] - np.dot(B[i, :], B[j, :].T) / (np.linalg.norm(B[j, :], ord=2) ** 2) * B[j, :]

        if self.space == "Real":
            for i in range(0, B.shape[0]):
                B[i, :] = B[i, :] / np.linalg.norm(B[i, :], ord=2)
            return B.T
        else:
            C = np.zeros([self.nql * self.num_observations * self.class_num, self.nql], dtype=complex).T
            for i in range(0, B.shape[0], 2):
                C[int(i / 2), :] = B[i, :] + 1j * B[i + 1, :]
                C[int(i / 2), :] = C[int(i / 2), :] / np.linalg.norm(C[int(i / 2), :], ord=2)
            return C.T
