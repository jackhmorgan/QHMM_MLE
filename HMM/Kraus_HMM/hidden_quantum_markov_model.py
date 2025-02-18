# 求解hidden quantum Markov model 和 split hidden quantum Markov model的代码
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from qiskit import *
from hidden_quantum_markov_model_setup import hidden_quantum_Markov_model_setup


class hidden_quantum_Markov_model(hidden_quantum_Markov_model_setup):

    def __init__(self, data, val_data, test_data, max_iter=10, tau=0.75, alpha=0.92, beta=0.9, batch_num=200
                 , len_each_batch=300, qubits=1, random_seed=100, output_dims=6, class_num=1, space="Complex",
                 random_seed_for_density=80):
        super(hidden_quantum_Markov_model, self).__init__(data, val_data, test_data, maxIter=max_iter,
                                                          tau=tau, alpha=alpha, beta=beta, batch_num=batch_num,
                                                          len_each_batch=len_each_batch,
                                                          qubits=qubits, random_seed=random_seed,
                                                          output_dims=output_dims, class_num=class_num, space=space,
                                                          random_seed_for_density=random_seed_for_density)

        rho_initial = self.Initial_Density_Matrix()
        self.rho_initial = rho_initial
        self.rho_initial_copy = rho_initial
        Kraus_initial = self.Generate_Kraus()
        self.Kraus_initial = Kraus_initial
        # 自动调用检查数据格式的函数
        self.check_initial()

    def check_initial(self):
        eps = 1e-5
        if np.trace(self.rho_initial) >= 1 + eps or np.trace(self.rho_initial) < 1 - eps or \
                self.dagger(self.rho_initial).any() != self.rho_initial.any():
            raise ValueError("密度矩阵初始化失败，初始化密度矩阵不满足条件！")
        if np.dot(self.dagger(self.Kraus_initial), self.Kraus_initial).any() != np.identity(2).any():
            raise ValueError("Kraus算符初始化失败，不满足Stiefel流形的条件！")

    @staticmethod
    def like_hood(Kraus, rho_initial=0, train_data_seq=0, burnin=100):
        """

        :param Kraus: Kraus算符矩阵
        :param rho_initial:初始化密度矩阵
        :param train_data_seq:训练数据
        :param burnin: 预处理的数据个数
        :return: 似然函数值
        """
        # burn in
        rho = 0
        for i in train_data_seq[0:burnin]:
            rho_new = np.dot(np.conj(Kraus[2 * i - 2:2 * i, :]),
                             np.dot(rho_initial, np.transpose(Kraus[2 * i - 2:2 * i, :])))
            rho_initial = rho_new / np.trace(rho_new)
        # calculate like-hood
        for j in train_data_seq[burnin:]:
            rho = np.dot(np.conj(Kraus[2 * j - 2:2 * j, :]),
                         np.dot(rho_initial, np.transpose(Kraus[2 * j - 2:2 * j, :])))
            rho_initial = rho
        p = np.trace(rho)
        return -np.log(np.real(p))

    def __compute_like_hood_grad_fun(self):
        self.grad_like_hood = egrad(self.like_hood, 0)
        self.grad_fun = self.grad_like_hood
        return self.grad_fun

    def compute_grad_of_like_hood(self, train_data_seq):
        func = self.__compute_like_hood_grad_fun()
        return func(np.conj(self.Kraus_initial), rho_initial=self.rho_initial, train_data_seq=train_data_seq,
                    burnin=100)
    
    def Iteration_Step(self):
        DA_value = []
        G_old = 0
        for i in range(self.maxIter):
            print('-----Iteration number : ', i)
            loss = []
            for j in range(self.batch_num):
                print(j)
                data_train_seq = self.data_Pre(mode="train")[j, :]
                G = self.compute_grad_of_like_hood(train_data_seq=data_train_seq)
                F = np.linalg.norm(G, ord=2)
                G = G / F
                G = self.beta * G_old + (1 - self.beta) * G
                E = np.linalg.norm(G, ord=2)
                G = G / E
                U = np.hstack((G, self.Kraus_initial))
                V = np.hstack((self.Kraus_initial, -G))
                Inverse = np.identity(2 * self.dim_K) + self.tau / 2 * np.dot(self.dagger(V), U)
                item1 = np.dot(U, np.linalg.inv(Inverse))
                item2 = np.dot(self.dagger(V), self.Kraus_initial)
                self.Kraus_initial = self.Kraus_initial - self.tau * np.dot(item1, item2)
                G_old = G
                L_train = self.like_hood(np.conj(self.Kraus_initial), rho_initial=self.rho_initial,
                                         train_data_seq=data_train_seq,
                                         burnin=100)
                loss.append(L_train)
            DA_value.append(self.DA(np.mean(loss)))
            self.tau = self.alpha * self.tau
        return DA_value, self.Kraus_initial
