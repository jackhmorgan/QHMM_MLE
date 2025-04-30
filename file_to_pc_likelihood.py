from HMM import QHMM, PC_HMM, NPC_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils import kl_divergence_between_hmm
import numpy as np
import json
import os
import argparse
import time

parser = argparse.ArgumentParser(description="Parse command line arguments")

# Argument 1: List of numbers
parser.add_argument(
    '--n_samples', 
    type=int,  # Convert each input to a float
    help='Number of repetitions',
)

parser.add_argument(
    '--len_sequence', 
    type=int,  # Convert each input to a float
    help='length of an individual sequence',
)

# Argument 3: String
parser.add_argument(
    '--path', 
    type=str,  # Expect a string
    help='A string argument',
)

# Argument 3: String
parser.add_argument(
    '--source_filename', 
    type=str,  # Expect a string
    help='A string argument',
)

parser.add_argument(
    '--ncl', 
    type=int,  # Convert each input to a float
    help='The number of classical latent states',
)

parser.add_argument(
    '--k', 
    type=int,  # Convert each input to a float
    help='The number of spot volatilities per integrated volatility',
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
path = args.path if args.path else 'MLE/ClassicalConvergence/pc_to_npc_to_qhmm_test.json'

source_filename = args.source_filename if args.source_filename else 'PC_to_NPC_to_QHMM/new_result_getter_16.json'

with open(folder+'/'+filename+'.json', 'r') as file:
    data = json.load(file)
    
len_sequence = args.len_sequence if args.len_sequence else 10
k = args.k if args.k else 4 
ncl = args.ncl if args.ncl else 4 
n_samples = args.n_samples if args.n_samples else 10
max_iter = args.max_iter if args.max_iter else 100
tol = args.tol if args.tol else 0.01

filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1
print(path)
# Define QHMM hyperparams

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, efficient_su2
from HMM.utils.qhmm_utils import statevector_result_getter

result_getter = statevector_result_getter()

# 2 qubits = 4 hidden states
initial_state = QuantumCircuit(2, name='Initial_State')
initial_state.h(0)
initial_state.cx(0,1)

ansatz = efficient_su2(4, reps=3, entanglement='full', su2_gates=['rz','ry'])

# initial_state = QuantumCircuit(1, name='Initial_State')
# initial_state.h(0)

# ansatz = efficient_su2(3, reps=3, entanglement='full', su2_gates=['rz','ry'])

transition_matrix = np.random.rand(ncl, ncl)
    
# Normalize each row so that it sums to 1
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

theta_gen = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
theta_0_npc = theta = np.array(transition_matrix)[:,:-1].flatten().tolist()
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]
alpha= np.log(observations[-1]-observations[0])
delta = -alpha/10
max_iter = 1000

theta_0_q  = [np.random.uniform(2*np.pi,  6* np.pi) for _ in range(ansatz.num_parameters)]

data = {}


data['gen_theta'] = theta_gen
data['theta_0_npc'] = theta_0_npc
data['theta_0_q'] = theta_0_q
data['len_sequence'] = len_sequence
data['ncl'] = ncl
data['k'] = k
        
with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)
    
for iteration in range(n_samples):
    print('here 1')
    model_1 = PC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_gen,
                      observations = observations)

    print('here 2')
    model_2  = NPC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_0_npc,
                      observations = observations,
                      alpha = alpha,
                      delta = delta)
    print('here 3')
    model_3 = QHMM(theta=theta_0_q,
                   result_getter=result_getter,
                   initial_state=initial_state,
                   ansatz=ansatz)
    print('here 4')
    #sequence = model_1.generate_sequence(10*(iteration+1))
    sequence = model_1.generate_sequence(len_sequence)
    
    print('here 5')
    model_1_likelihood = model_1.log_likelihood(sequence)
    print('here 6')
    model_2_likelihood = 1 # model_2.log_likelihood(sequence)
    print('here 7')
    model_3_likelihood = 1 #model_3.log_likelihood(sequence)
    print('here 8')
    
    theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(model = model_3,
              sequence=sequence,
              theta_0=theta_0_q,
              max_iter=max_iter,
              tol=tol)
    print('here 9')
    theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(model = model_2,
              sequence=sequence,
              theta_0=theta_0_npc,
              max_iter=max_iter,
              tol=tol)
    print('here 10')
    quantum_kl = 'na' #float(kl_divergence_between_hmm(model_1, model_3, n_sequences=10))
    npc_kl = 'na' #float(kl_divergence_between_hmm(model_1, model_2, n_sequences=10))
    print('here 11')
    n_samples_data = {'quantum_kl' : quantum_kl,
                      'npc_kl' : npc_kl,
                      'model_1_likelihood' : model_1_likelihood,
                      'model_2_likelihood' : model_2_likelihood,
                      'model_3_likelihood' : model_3_likelihood,
                      'trained_q_likelihood' : float(np.exp(-training_curve_q[-1])),
                      'trained_npc_likelihood' : float(np.exp(-training_curve_npc[-1])),
                      'theta_trained_q' : theta_trained_q,  
                      'training_time_q': training_time_q,
                      'nit_q' : nit_q,
                      'training_curve_q': training_curve_q,
                      'nit_npc' : nit_npc,
                      'theta_trained_npc' : theta_trained_npc,  
                      'training_time_npc': training_time_npc,
                      'training_curve_npc': training_curve_npc,
                      'sequence' : sequence.tolist(),
                      }
    data[iteration] = n_samples_data
    
    with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)