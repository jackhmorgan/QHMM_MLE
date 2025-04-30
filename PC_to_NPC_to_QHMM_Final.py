'''
Copyright 2025 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from HMM import QHMM, PC_HMM, NPC_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
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
    help='A list of numbers (space-separated)',
)

# Argument 3: String
parser.add_argument(
    '--path', 
    type=str,  # Expect a string
    help='A string argument',
)

parser.add_argument(
    '--len_sequence', 
    type=int,  # Convert each input to a float
    help='The length of each individual sample',
)

parser.add_argument(
    '--k', 
    type=int,  # Convert each input to a float
    help='The number of spot volatilities per integrated volatility',
)

parser.add_argument(
    '--ncl', 
    type=int,  # Convert each input to a float
    help='The number of classical latent states',
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
path = args.path if args.path else 'MLE/ClassicalConvergence/pc_to_npc_to_qhmm_test.json'
len_sequence = args.len_sequence if args.len_sequence else 500 # 16 parameters x 100 
n_samples = args.n_samples if args.n_samples else 100
max_iter = args.max_iter if args.max_iter else 1000
tol = args.tol if args.tol else 0.0001

k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4

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


initial_state = QuantumCircuit(1, name='Initial_State')
initial_state.h(0)

ansatz = efficient_su2(3, reps=3, entanglement='full', su2_gates=['rz','ry'])

#initial_state = QuantumCircuit(2, name='Initial_State')
#initial_state.h(0)
#initial_state.cx(0,1)

#ansatz = efficient_su2(4, reps=3, entanglement='full', su2_gates=['rz','ry'])

theta_gen = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]
max_iter = 1000

data = {}


data['gen_theta'] = theta_gen
data['len_sequence'] = len_sequence
data['ncl'] = ncl
data['k'] = k

with open(path, "w") as outfile: 
    json.dump(data, outfile, indent=4)

for sample in range(n_samples):
    theta_0_q  = [np.random.uniform(2*np.pi,  6* np.pi) for _ in range(ansatz.num_parameters)]
    transition_matrix = np.random.rand(ncl, ncl)
    
    # Normalize each row so that it sums to 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    theta_0_npc = theta = np.array(transition_matrix)[:,:-1].flatten().tolist()
    
    model_1 = PC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_gen,
                      observations = observations)

    model_2  = NPC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_0_npc,
                      observations = observations)
    
    model_3 = QHMM(theta=theta_0_q,
                   result_getter=result_getter,
                   initial_state=initial_state,
                   ansatz=ansatz)

    sequence = model_1.generate_sequence(n_samples)

    model_1_likelihood = model_1.log_likelihood(sequence)
    model_2_likelihood = 1 #model_2.log_likelihood(sequence)
    model_3_likelihood = 1 #model_3.log_likelihood(sequence)
    
    theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(model = model_3,
              sequence=sequence,
              theta_0=theta_0_q,
              max_iter=max_iter,
              tol=tol)
    
    theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(model = model_2,
              sequence=sequence,
              theta_0=theta_0_npc,
              max_iter=max_iter,
              tol=tol)
    
    n_samples_data = {'model_1_likelihood' : model_1_likelihood,
                      'model_2_likelihood' : model_2_likelihood,
                      'model_3_likelihood' : model_3_likelihood,
                      'trained_q_likelihood' : np.exp(-training_curve_q[-1]),
                      'trained_npc_likelihood' : np.exp(-training_curve_npc[-1]),
                      'theta_trained_q' : theta_trained_q,  
                      'training_time_q': training_time_q,
                      'nit_q' : nit_q,
                      'training_curve_q': training_curve_q,
                      'nit_npc' : nit_npc,
                      'theta_trained_npc' : theta_trained_npc,  
                      'training_time_npc': training_time_npc,
                      'training_curve_npc': training_curve_npc,
                      'sequence' : sequence.tolist()
                      }
    data[samples] = n_samples_data
    
    with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)