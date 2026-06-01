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

from HMM import QHMM, PC_HMM, NPC_HMM, GARCH
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils.pc_utils import minimize_pc_hmm
from HMM.utils.garch_utils import minimize_GARCH
import pandas as pd
import numpy as np
import json
import os
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Parse command line arguments")

parser.add_argument(
    '--n_samples', 
    type=int,
    help='Number of sequences to generate and test',
)

parser.add_argument(
    '--garch_results_path', 
    type=str,
    help='Path to the GARCH training results JSON file',
)

parser.add_argument(
    '--output_path', 
    type=str,
    help='Path to save comparison results',
)

parser.add_argument(
    '--len_sequence', 
    type=int,
    help='The length of each individual sample',
)

parser.add_argument(
    '--k', 
    type=int,
    help='The number of spot volatilities per integrated volatility',
)

parser.add_argument(
    '--ncl', 
    type=int,
    help='The number of classical latent states',
)

parser.add_argument(
    '--max_iter',
    type=int,
    help='The maximum number of optimization iterations',
)

parser.add_argument(
    '--tol',
    type=float,
    help='The improvement tolerance to end convergence',
)

args = parser.parse_args()

# Set defaults
output_path = args.output_path if args.output_path else 'MLE/garch_to_pc_to_npc_to_qhmm.json'
len_sequence = args.len_sequence if args.len_sequence else 500
n_samples = args.n_samples if args.n_samples else 10
max_iter = args.max_iter if args.max_iter else 1000
tol = args.tol if args.tol else 0.0001
k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4

# ============================================================================
# HARD-CODED PARAMETERS - EDIT ONLY THE TWO LINES BELOW
# ============================================================================
garch_p = 1  # Typical SPY GARCH AR order
garch_q = 1  # Typical SPY GARCH MA order

# Handle output path with auto-increment
filename, extension = os.path.splitext(output_path)
counter = 1

while os.path.exists(output_path):
    output_path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# Load SPY data for GARCH sequence generation
df = pd.read_csv('^SPX.csv')
log_returns = pd.DataFrame({'log_returns': np.log(df['Close'].shift(-1) / df['Close'])})
log_returns = log_returns.dropna().reset_index(drop=True)
log_returns_values = log_returns['log_returns'].values

garch_model = {
                "omega": 2.5896387874042265e-06,
                "alpha[1]": 0.1,
                "alpha[2]": 0.1,
                "beta[1]": 0.26,
                "beta[2]": 0.26,
                "beta[3]": 0.26
            }
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]


# Setup QHMM parameters
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from HMM.utils.qhmm_utils import statevector_result_getter

result_getter = statevector_result_getter(rescaling_factor=1e6)

initial_state = QuantumCircuit(1, name='Initial_State')
initial_state.h(0)

ansatz = efficient_su2(3, reps=3, entanglement='full', su2_gates=['rz','ry'])

# Initialize output data structure
output_data = {}
output_data['garch_model'] = garch_model
output_data['observations'] = observations
output_data['len_sequence'] = len_sequence
output_data['ncl'] = ncl
output_data['k'] = k
output_data['n_samples'] = n_samples

with open(output_path, "w") as outfile: 
    json.dump(output_data, outfile, indent=4)

# Generate GARCH sequences and train models
print(f"\nGenerating {n_samples} sequences and training models...")

for sample in range(n_samples):
    print(f"\n--- Sample {sample + 1}/{n_samples} ---")
    
    # Use hard-coded theta for PC_HMM
    theta_gen_pc = theta_gen
    
    # Generate random initial theta for NPC_HMM
    transition_matrix = np.random.rand(ncl, ncl)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    theta_0_npc = np.array(transition_matrix)[:,:-1].flatten().tolist()
    
    # Generate random initial theta for QHMM
    theta_0_q = [np.random.uniform(2*np.pi, 6*np.pi) for _ in range(ansatz.num_parameters)]
    
    # Create models
    model_garch = GARCH(p=best_garch['p'], 
                       q=best_garch['q'],
                       observations=log_returns_values)
    
    model_pc = PC_HMM(k=k,
                      ncl=ncl,
                      theta=theta_gen_pc,
                      observations=observations)
    
    model_npc = NPC_HMM(k=k,
                       ncl=ncl,
                       theta=theta_0_npc,
                       observations=observations)
    
    model_qhmm = QHMM(theta=theta_0_q,
                      result_getter=result_getter,
                      initial_state=initial_state,
                      ansatz=ansatz)
    
    # Generate sequence from GARCH (use actual log returns as a realistic sequence)
    # Sample with replacement from the log returns
    sequence_indices = np.random.choice(len(log_returns_values), len_sequence, replace=True)
    sequence = log_returns_values[sequence_indices].reshape(-1, 1)
    
    # Discretize sequence into observation bins for PC_HMM and NPC_HMM
    bins = [log_returns_values.quantile((i+1)/no) for i in range(no-1)]
    discrete_sequence = []
    for lr in sequence.flatten():
        for i, edge in enumerate(bins):
            if lr <= edge:
                discrete_sequence.append(i)
                break
        else:
            discrete_sequence.append(len(bins))
    
    discrete_sequence = np.array(discrete_sequence).reshape(-1, 1)
    
    try:
        # Calculate GARCH likelihood
        garch_likelihood = model_garch.log_likelihood(sequence.flatten())
    except:
        garch_likelihood = -np.inf
    
    try:
        # Calculate PC_HMM likelihood
        pc_likelihood = model_pc.log_likelihood(discrete_sequence)
    except:
        pc_likelihood = -np.inf
    
    print(f"Initial likelihoods - GARCH: {garch_likelihood:.4f}, PC: {pc_likelihood:.4f}")
    
    # Train PC_HMM
    try:
        print("Training PC_HMM...")
        theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(
            model=model_pc,
            sequence=discrete_sequence,
            theta_0=theta_gen_pc,
            max_iter=max_iter,
            tol=tol)
        trained_pc_likelihood = np.exp(-training_curve_pc[-1]) if training_curve_pc else -np.inf
        print(f"PC_HMM trained - Final LL: {training_curve_pc[-1]:.4f}")
    except Exception as e:
        print(f"PC_HMM training failed: {str(e)}")
        theta_trained_pc = None
        training_time_pc = 0
        nit_pc = 0
        training_curve_pc = []
        trained_pc_likelihood = -np.inf
    
    # Train NPC_HMM
    try:
        print("Training NPC_HMM...")
        theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(
            model=model_npc,
            sequence=discrete_sequence,
            theta_0=theta_0_npc,
            max_iter=max_iter,
            tol=tol)
        trained_npc_likelihood = np.exp(-training_curve_npc[-1]) if training_curve_npc else -np.inf
        print(f"NPC_HMM trained - Final LL: {training_curve_npc[-1]:.4f}")
    except Exception as e:
        print(f"NPC_HMM training failed: {str(e)}")
        theta_trained_npc = None
        training_time_npc = 0
        nit_npc = 0
        training_curve_npc = []
        trained_npc_likelihood = -np.inf
    
    # Train QHMM
    try:
        print("Training QHMM...")
        theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(
            model=model_qhmm,
            sequence=discrete_sequence,
            theta_0=theta_0_q,
            max_iter=max_iter,
            tol=tol)
        trained_qhmm_likelihood = np.exp(-training_curve_q[-1]) if training_curve_q else -np.inf
        print(f"QHMM trained - Final LL: {training_curve_q[-1]:.4f}")
    except Exception as e:
        print(f"QHMM training failed: {str(e)}")
        theta_trained_q = None
        training_time_q = 0
        nit_q = 0
        training_curve_q = []
        trained_qhmm_likelihood = -np.inf
    
    # Save results for this sample
    sample_data = {
        'garch_likelihood': float(garch_likelihood),
        'initial_pc_likelihood': float(pc_likelihood),
        'initial_theta_pc': theta_gen_pc,
        'trained_pc_likelihood': float(trained_pc_likelihood),
        'theta_trained_pc': theta_trained_pc,
        'training_time_pc': training_time_pc,
        'nit_pc': nit_pc,
        'training_curve_pc': training_curve_pc,
        'initial_theta_npc': theta_0_npc,
        'trained_npc_likelihood': float(trained_npc_likelihood),
        'theta_trained_npc': theta_trained_npc,
        'training_time_npc': training_time_npc,
        'nit_npc': nit_npc,
        'training_curve_npc': training_curve_npc,
        'initial_theta_qhmm': theta_0_q,
        'trained_qhmm_likelihood': float(trained_qhmm_likelihood),
        'theta_trained_qhmm': theta_trained_q,
        'training_time_qhmm': training_time_q,
        'nit_qhmm': nit_q,
        'training_curve_qhmm': training_curve_q,
    }
    
    output_data[sample] = sample_data
    
    with open(output_path, "w") as outfile: 
        json.dump(output_data, outfile, indent=4, default=str)

print(f"\n\nResults saved to: {output_path}")
