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
from HMM.utils.pc_utils import minimize_pc_hmm
from HMM.utils import kl_divergence_between_hmm
import numpy as np
import numpy.random as random
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
    '--max_l_bits', 
    type=int,  # Convert each input to a float
    help='The number of latent qubits',
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
path = args.path if args.path else 'MLE/ClassicalConvergence_with_pc/pc_to_pc_to_npc_to_qhmm_test.json'
len_sequence = args.len_sequence if args.len_sequence else 10
max_l_bits = args.max_l_bits if args.max_l_bits else 4 
ncl = args.ncl if args.ncl else 4 
k = args.k if args.k else 4 
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


initial_states = []
ansatz_circuits = []

for i in range(1, max_l_bits+1):
    i_s = QuantumCircuit(i, name='Initial_State')
    i_s.h(0)
    for j in range(1, i):
        i_s.cx(j-1, j)
    initial_states.append(i_s)
    
    ansatz = efficient_su2(i+2, reps=3, entanglement='full', su2_gates=['rz','ry'])
    ansatz_circuits.append(ansatz)

theta_gen = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]
max_iter = 100

theta_list = [[np.random.uniform(2*np.pi,  6* np.pi) for _ in range(ansatz.num_parameters)] for ansatz in ansatz_circuits]
        
data = {}

data['theta_list'] = theta_list
data['len_sequence'] = len_sequence
        
with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)
    
for iteration in range(n_samples):
    data[iteration] = {}
    print('here 1')
    model_1 = PC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_gen,
                      observations = observations)
    quantum_models = []
    for i in range(len(theta_list)):
        quantum_models.append(QHMM(theta=theta_list[i],
                   result_getter=result_getter,
                   initial_state=initial_states[i],
                   ansatz=ansatz_circuits[i]))

    #sequence = model_1.generate_sequence(10*(iteration+1))
    sequence = model_1.generate_sequence(len_sequence)
    pc_likelihood = model_1.log_likelihood(sequence)
    
    for i, model_q in enumerate(quantum_models):
        print('here 8')
        theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(model = model_q,
                                                                                  theta_0 = theta_list[i],
                                                                                  sequence=sequence,
                                                                                  max_iter=max_iter,
                                                                                  tol=tol)
        print('here 9')


     
        n_samples_data = {'trained_q_likelihood' : float(np.exp(-training_curve_q[-1])),
                          'theta_trained_q' : theta_trained_q,  
                          'training_time_q': training_time_q,
                          'nit_q' : nit_q,
                          'training_curve_q': training_curve_q,
                          'sequence' : sequence.tolist(),
                          'pc_likelihood' : float(pc_likelihood),
                          }
        data[iteration][i] = n_samples_data
    
        with open(path, "w") as outfile: 
            json.dump(data, outfile, indent=4)