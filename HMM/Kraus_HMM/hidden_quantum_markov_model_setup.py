# 求解hidden quantum Markov model 和 split hidden quantum Markov model的代码
import autograd.numpy as np
from numpy.linalg import matrix_rank as rank
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace
from autograd.numpy import pi

from qiskit_aer import AerSimulator


class hidden_quantum_Markov_model_setup:
    """
    hidden_quantum_Markov_model_setup
    data, vla_data, test_data,必须全部是待训练的整型数据
    maxIter:最大迭代步数，默认取值是10，可以调整，必须是整型
    tau:学习率因子，默认是0.95，可以调整必须是0-1之间的小数
    alpha:学习率衰减因子，默认是0.95，可以调整必须是0-1之间的小数
    beta:记忆因子，记忆上次计算的梯度值在本次计算中应该保留多少，默认是0.90，可以调整，必须是0-1之间的小数
    batch_num:训练数据的batch数目，可以调整，整型数据
    batch_num_val:验证数据的batch数目，可以调整，整型数据
    batch_num_test:测试数据的batch数据，可以调整，整型数据
    len_each_batch:每一个batch所对应的数据长度，整型
    qubits:需要的计算比特数目，目前只支持1比特和2比特的计算，整型
    random_seed:初始化Kraus算符所需要的随机数种子，默认值为float
    output_dims:模型输出的状态空间的维度，只能取整型
    class_num:Kraus算符的种类，整型数据
    space:生成的Kraus算符所在的空间，只能取值为Complex或Real，默认值是Complex
    """

    def __init__(self, data, val_data, test_data, maxIter=10, tau=0.95, alpha=0.95, beta=0.90,
                 batch_num=None, batch_num_val=None,
                 batch_num_test=None, len_each_batch=None, qubits=1, random_seed=None, output_dims=None, class_num=None,
                 space="Complex", random_seed_for_density=None):
        self.data = data
        self.val_data = val_data
        self.test_data = test_data
        self.maxIter = maxIter
        self.tau = tau
        self.tau_copy = tau
        self.alpha = alpha
        self.beta = beta
        self.batch_num = batch_num
        self.batch_num_val = batch_num_val
        self.batch_num_test = batch_num_test
        self.len_each_batch = len_each_batch
        self.qubits = qubits
        self.dim_K = 2 ** qubits
        self.random_seed = random_seed
        self.output_dims = output_dims
        self.class_num = class_num
        self.space = space
        self.random_seed_for_density = random_seed_for_density
        self.check_data()

    def data_Pre(self, mode="train"):
        """
        预处理输入的数据，把数据重新改变大小为 batch数 * 长度
        :param mode: 可选参数，"train","validation","test"分别对应预处理训练数据，验证数据，测试数据
        :return: 预处理后的数据
        """
        if mode == "train":
            train_seq = self.data.reshape(self.batch_num, self.len_each_batch)
            return train_seq
        if mode == "validation":
            val_seq = self.val_data.reshape(self.batch_num_val, self.len_each_batch)
            return val_seq
        if mode == "test":
            test_seq = self.test_data.reshape(self.batch_num_test, self.len_each_batch)
            return test_seq

    def check_data(self):
        """
        检查数据类型是否正确，目前只能处理离散的数据，所以数据类型只能取int型，根据机器的不同int32或int64
        :return: None
        """
        if self.data.dtype != "int64":
            raise TypeError("待训练的数据必须是整型")

    @staticmethod
    def dagger(M):
        """
        计算矩阵M的dagger：先进行元素共轭，然后进行转置操作
        :param M: 矩阵
        :return: 矩阵M的dagger
        """
        return np.transpose(np.conj(M))

    def DA(self, L_train, seq_len=200):
        """
        计算模型的DA值
        :param L_train:计算出来的似然函数值
        :param seq_len: 带入训练的每个batch的序列长度再减去burnin的长度
        :return: 似然函数值
        """

        def fun(t):
            if t > 0:
                return t
            else:
                f = (1 - np.exp(-0.25 * t)) / (1 + np.exp(-0.25 * t))
            return f

        z = fun(1 - L_train / (np.log(self.output_dims) * seq_len))
        return z

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

    @staticmethod
    def KMR(Kraus, rho):
        "计算Kraus算符夹密度矩阵"
        z = np.dot(np.conj(Kraus), np.dot(rho, np.transpose(Kraus)))
        return z

    def Generate_Kraus(self):
        """
        产生初始的的Kraus算符
        :return:
        """

        np.random.seed(self.random_seed)
        if self.space == "Real":
            A = np.random.random([self.dim_K * self.output_dims * self.class_num, self.dim_K])
        elif self.space == "Complex":
            A = np.random.random([self.dim_K * self.output_dims * self.class_num, self.dim_K * 2])
        else:
            raise ValueError("参数space只能接受Real或者Complex")
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
            C = np.zeros([self.dim_K * self.output_dims * self.class_num, self.dim_K], dtype=complex).T
            for i in range(0, B.shape[0], 2):
                C[int(i / 2), :] = B[i, :] + 1j * B[i + 1, :]
                C[int(i / 2), :] = C[int(i / 2), :] / np.linalg.norm(C[int(i / 2), :], ord=2)
            return C.T

    def Initial_Density_Matrix(self):
        """
        用量子线路产生初始的密度矩阵，目前只能产生1比特和2比特的密度矩阵
        :return: 密度矩阵
        """
        theta0, theta1, theta2 = Parameter('θ0'), Parameter('θ1'), Parameter('Θ2')
        qr = QuantumRegister(4, 'q')
        qc = QuantumCircuit(qr)
        for i in range(4): qc.h(qr[i])
        qc.crx(theta0, qr[0], qr[1])
        qc.cry(theta1, qr[1], qr[2])
        qc.crz(theta2, qr[2], qr[3])
        #np.random.seed(self.random_seed_for_density)
        parameters = np.random.rand(3)
        parameters = 2 * pi * parameters
        qc.assign_parameters(parameters=parameters.tolist(), inplace=True)
        qc.save_state()
        backend = AerSimulator(method='statevector')
        job = backend.run(qc)
        value = job.result().data()['statevector']
        if self.dim_K == 2:
            subsystem_density_matrix = partial_trace(value, qargs=[0, 1, 2])
        elif self.dim_K == 4:
            subsystem_density_matrix = partial_trace(value, qargs=[0, 1])
        elif self.dim_K == 8:
            subsystem_density_matrix = partial_trace(value, qargs=[0])

        else:
            raise ValueError("当前输入的比特只支持单比特计算和双比特")
        return subsystem_density_matrix._data