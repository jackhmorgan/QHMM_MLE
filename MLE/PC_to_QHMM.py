from HMM import QHMM, PC_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.pc_utils import minimize_pc_hmm
import numpy as np
import json
import os
import argparse
import time

parser = argparse.ArgumentParser(description="Parse command line arguments")

# Argument 1: List of numbers
parser.add_argument(
    '--n_samples_list', 
    nargs='+',  # Allow multiple arguments
    type=int,  # Convert each input to a float
    help='A list of numbers (space-separated)',
)

# Argument 3: String
parser.add_argument(
    '--path', 
    type=str,  # Expect a string
    help='A string argument',
)

parser.add_argument(
    '--sample_length', 
    type=float,  # Convert each input to a float
    help='The length of each individual sample',
)

parser.add_argument(
    '--max_iter',
    type=int,
    help='The maximum number of optimiation iterations',
)

parser.add_argument(
    '--tol',
    type=float,
    help='The improvement tolerance to end convergence',
)

args = parser.parse_args()
# Determine the number of time steps in our sample size
path = args.path if args.path else 'MLE/ClassicalConvergence/pc_to_qhmm_test.json'
sample_length = args.sample_length if args.sample_length else 1600 # 16 parameters x 100 
n_samples_list = args.n_samples_list if args.n_samples_list else [10]
max_iter = args.max_iter if args.max_iter else 100
tol = args.tol if args.tol else 0.0001

filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# Define QHMM hyperparams

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, efficient_su2
from HMM.utils.qhmm_utils import statevector_result_getter

result_getter = statevector_result_getter()

# 2 qubits = 4 hidden states
initial_state = QuantumCircuit(2, name='Initial_State')
initial_state.h(0)
initial_state.cx(0,1)

ansatz = real_amplitudes(4, reps=1, entanglement='pairwise', )
#ansatz = efficient_su2(4, reps=1, entanglement='pairwise', su2_gates=['rz','ry'])

ncl = 16
theta_gen = [0.2, 0.05, 0.2]
theta_0_pc = [0.1, 0.1, 0.1]
k = 4
observations = [-0.3, -0.1, 0.1, 0.3]
max_iter = 100

theta_0_q  = [np.random.uniform(2*np.pi,  6* np.pi) for _ in range(ansatz.num_parameters)]

data = {}


data['gen_theta'] = theta_gen
data['theta_0_pc'] = theta_0_pc
data['theta_0_q'] = theta_0_q
        
for n_samples in n_samples_list:
    
    model_1 = PC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_gen,
                      observations = observations)

    model_2  = PC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_0_pc,
                      observations = observations)
    
    model_3 = QHMM(theta=theta_0_q,
                   result_getter=result_getter,
                   initial_state=initial_state,
                   ansatz=ansatz)

    sequence = model_1.generate_sequence(n_samples)

    model_1_likelihood = model_1.log_likelihood(sequence)
    model_2_likelihood = model_2.log_likelihood(sequence)
    model_3_likelihood = model_3.log_likelihood(sequence)
    
    theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(model = model_3,
              sequence=sequence,
              theta_0=theta_0_q,
              max_iter=max_iter,
              tol=tol)
    
    theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(model = model_2,
              sequence=sequence,
              theta_0=theta_0_pc,
              max_iter=max_iter,
              tol=tol)
    
    n_samples_data = {'model_1_likelihood' : model_1_likelihood,
                      'model_2_likelihood' : model_2_likelihood,
                      'model_3_likelihood' : model_3_likelihood,
                      'trained_q_likelihood' : np.exp(-training_curve_q[-1]),
                      'trained_pc_likelihood' : np.exp(-training_curve_pc[-1]),
                      'theta_trained_q' : theta_trained_q,  
                      'training_time_q': training_time_q,
                      'nit_q' : nit_q,
                      'training_curve_q': training_curve_q,
                      'nit_pc' : nit_pc,
                      'theta_trained_np' : theta_trained_pc,  
                      'training_time_np': training_time_pc,
                      'training_curve_np': training_curve_pc,
                      }
    data[n_samples] = n_samples_data
    
    with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)