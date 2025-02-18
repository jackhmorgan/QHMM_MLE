import autograd.numpy as np
#from HQMM_Backup import split_hidden_quantum_Markov_model as shqmm
from hidden_quantum_markov_model import hidden_quantum_Markov_model as hqmm
import os
import sys

import autograd

def getMaxPosition(M):
    max_value = max(M)
    for i in range(M.shape[0]):
        if M[i] == max_value:
            return max_value, i




def LodaData(datatype="quantum"):
    if datatype == "quantum":
        train_data = np.loadtxt('HMM\Kraus_HMM\seq_hmm_test.txt')
        train_data = train_data.astype(int)
        val_data = np.loadtxt('HMM\Kraus_HMM\\val_sequence.txt')
        val_data = val_data.astype(int)
        test_data = np.loadtxt('HMM\Kraus_HMM\\test_sequence.txt')
        test_data = test_data.astype(int)
        return train_data, val_data, test_data
    elif datatype == "classical":
        train_data = np.loadtxt("HMM\Kraus_HMM\seq_hmm_train.txt")
        train_data = train_data.astype(int)
        val_data = np.loadtxt("HMM\Kraus_HMM\seq_hmm_val.txt")
        val_data = val_data.astype(int)
        test_data = np.loadtxt("HMM\Kraus_HMM\seq_hmm_test.txt")
        test_data = test_data.astype(int)
        return train_data, val_data, test_data


str ="quantum"
train_data, val_data, test_data = LodaData(datatype=str)
# -------------------split hidden quantum markov model-------------------------
model = hqmm(train_data, val_data, test_data, max_iter=20, tau=0.95, alpha=0.95, beta=0.90,
              batch_num=100, len_each_batch=300, qubits=1, output_dims=6, class_num=3, random_seed=10000)
print(model.Kraus_initial)
DA_value, kraus_final = model.Iteration_Step()
print(DA_value)
print(kraus_final)

if str == "quantum":
    if 'win' in sys.platform:
        if os.path.exists("DA_model_quantum"):
            np.savetxt("DA_model_quantum\\DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_quantum\\DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
        else:
            os.mkdir("DA_model_quantum")
            np.savetxt("DA_model_quantum\\DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value)
            np.savetxt("DA_model_quantum\\DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
    if 'linux' in sys.platform:
        if os.path.exists("DA_model_quantum"):
            np.savetxt("DA_model_quantum/DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_quantum/DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
        else:
            os.mkdir("DA_model_quantum")
            np.savetxt("DA_model_quantum/DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_quantum/DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)

else:
    if 'win' in sys.platform:
        if os.path.exists("DA_model_classical"):
            np.savetxt("DA_model_classical\\DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_classical\\DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
        else:
            os.mkdir("DA_model_classical")
            np.savetxt("DA_model_classical\\DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_classical\\DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
    if 'linux' in sys.platform:
        if os.path.exists("DA_model_classical"):
            np.savetxt("DA_model_classical/DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_classical/DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)
        else:
            os.mkdir("DA_model_classical")
            np.savetxt("DA_model_classical/DA_trained %s%s%s" % (model.class_num, model.IP, model.qubits), DA_value_model)
            np.savetxt("DA_model_classical/DA_validation %s%s%s"%(model.class_num, model.IP, model.qubits), DA_val)