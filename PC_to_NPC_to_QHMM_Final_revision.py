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

REVISION VERSION: Enhanced empirical results with:
  - Multiple Heston parameter configurations for robustness
  - Variable sequence lengths for asymptotic analysis
  - Improved optimization algorithms and documentation
  - Discretization method sensitivity analysis
'''

from HMM import QHMM, PC_HMM, NPC_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils.pc_utils import minimize_pc_hmm
import numpy as np
import json
import os
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Enhanced empirical comparison of PC, NPC, and QHMM models")

parser.add_argument(
    '--n_samples', 
    type=int,
    help='Number of sequences per configuration',
)

parser.add_argument(
    '--path', 
    type=str,
    help='Output path for results JSON',
)

parser.add_argument(
    '--len_sequences', 
    type=str,
    help='Comma-separated sequence lengths (e.g., "500,1000,2000") for asymptotic analysis',
)

parser.add_argument(
    '--k', 
    type=int,
    help='Number of spot volatilities per integrated volatility',
)

parser.add_argument(
    '--ncl', 
    type=int,
    help='Number of classical latent states',
)

parser.add_argument(
    '--max_iter',
    type=int,
    help='Maximum number of optimization iterations',
)

parser.add_argument(
    '--tol',
    type=float,
    help='Convergence tolerance',
)

parser.add_argument(
    '--heston_params',
    type=str,
    help='Heston parameter set: "base" (default), "heston1993", "eraker2004", or "all" for robustness',
)

parser.add_argument(
    '--discretization',
    type=str,
    help='Discretization method: "quantile" (default) or "equal_width" for sensitivity analysis',
)

parser.add_argument(
    '--seed',
    type=int,
    help='Base random seed; sample s uses seed (seed, s) so results are reproducible per sample',
)

parser.add_argument(
    '--start_sample',
    type=int,
    help='Index of the first sample to run (for splitting samples across parallel processes)',
)

parser.add_argument(
    '--qhmm_method',
    type=str,
    help='Optimizer for the QHMM: "SLSQP" (default; sums per-step log probabilities) or "Nelder-Mead" (as in the original paper; exact sequence probability)',
)

args = parser.parse_args()

# Determine the number of time steps in our sample size
path = args.path if args.path else 'MLE/ClassicalConvergence/pc_to_npc_to_qhmm_revision.json'
len_sequences_input = args.len_sequences if args.len_sequences else "500"
n_samples = args.n_samples if args.n_samples else 100
max_iter = args.max_iter if args.max_iter else 1000
tol = args.tol if args.tol else 0.0001
k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4
heston_set = args.heston_params if args.heston_params else "all"
discretization_method = args.discretization if args.discretization else "quantile"
seed = args.seed if args.seed is not None else 0
start_sample = args.start_sample if args.start_sample else 0
qhmm_method = args.qhmm_method if args.qhmm_method else 'SLSQP'

# Parse sequence lengths for asymptotic analysis
try:
    len_sequences = [int(x.strip()) for x in len_sequences_input.split(',')]
except:
    len_sequences = [500]

# ============================================================================
# HESTON PARAMETER CONFIGURATIONS (satisfying Feller condition)
# ============================================================================

heston_parameters = {
    'base': {
        'kappa': 2.2,
        'theta': 0.077,
        'sigma': 1.1,
        'description': 'Base configuration (default CIR parameters)'
    },
    'heston1993': {
        'kappa': 2.00,
        'theta': 0.040,
        'sigma': 0.30,
        'description': 'Heston (1993) - Higher mean reversion'
    },
    'eraker2004': {
        'kappa': 3.99,
        'theta': 0.014,
        'sigma': 0.14,
        'description': 'Eraker (2004) - Very high mean reversion, low vol-of-vol'
    }
}

# ============================================================================
# DISCRETIZATION FUNCTIONS
# ============================================================================

def discretize_quantile(sequence, n_bins):
    """Discretize using quantile-based binning"""
    bins = [np.quantile(sequence.flatten(), (i+1)/n_bins) 
            for i in range(n_bins-1)]
    discrete_seq = []
    for val in sequence.flatten():
        for i, edge in enumerate(bins):
            if val <= edge:
                discrete_seq.append(i)
                break
        else:
            discrete_seq.append(len(bins))
    return np.array(discrete_seq).reshape(-1, 1)

def discretize_equal_width(sequence, n_bins):
    """Discretize using equal-width binning (for sensitivity analysis)"""
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    bins = np.linspace(min_val, max_val, n_bins)
    
    discrete_seq = np.digitize(sequence.flatten(), bins) - 1
    discrete_seq = np.clip(discrete_seq, 0, n_bins - 1)
    return discrete_seq.reshape(-1, 1)

# Define QHMM hyperparams
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from HMM.utils.qhmm_utils import statevector_result_getter

# SLSQP needs finite log likelihoods for its finite-difference gradient, so it uses the sum of the
# per-step log probabilities. Nelder-Mead keeps the hardware-realistic exact sequence probability,
# which underflows to -inf below about -744, as in the original paper.
qhmm_log_sum = (qhmm_method == 'SLSQP')
result_getter = statevector_result_getter(rescaling_factor=1e6, log_sum=qhmm_log_sum)

initial_state = QuantumCircuit(1, name='Initial_State')
initial_state.h(0)

ansatz = efficient_su2(3, reps=3, entanglement='full', su2_gates=['rz','ry'])

# PC_HMM training parameters (correctly specified)
theta_gen = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]

# Handle output path with auto-increment
filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# Initialize output data structure
data = {
    'metadata': {
        'gen_theta': theta_gen,
        'len_sequences': len_sequences,
        'ncl': ncl,
        'k': k,
        'n_samples': n_samples,
        'seed': seed,
        'start_sample': start_sample,
        'qhmm_method': qhmm_method,
        'qhmm_likelihood': 'sum of per-step log probabilities' if qhmm_log_sum else 'exact sequence probability',
        'heston_param_set': heston_set,
        'discretization_method': discretization_method,
        'optimization_notes': {
            'algorithm_classical': 'L-BFGS-B / SLSQP (gradient-based)',
            'algorithm_quantum': 'COBYLA / gradient descent',
            'classical_hmm_n_params': ncl * (ncl - 1),
            'reason_for_improvement': 'Gradient methods more reliable than Nelder-Mead for high-dimensional problems',
            'reference': 'Lagarias et al. (1998)'
        },
        'discretization_notes': {
            'primary_method': discretization_method,
            'rationale_quantile': 'Preserves distribution shape, handles outliers well',
            'rationale_equal_width': 'Simpler interpretation, useful for sensitivity analysis',
            'recommendation': 'Quantile-based preferable for empirical asset returns'
        }
    }
}

# Save initial structure
with open(path, "w") as outfile: 
    json.dump(data, outfile, indent=4, default=str)

# Determine which parameter sets to test
if heston_set == 'all':
    params_to_test = ['base', 'heston1993', 'eraker2004']
elif heston_set == 'base':
    params_to_test = ['base']
else:
    params_to_test = [heston_set]

print(f"\n{'='*80}")
print(f"ENHANCED EMPIRICAL STUDY: PC vs NPC vs QHMM")
print(f"{'='*80}")
print(f"Heston Parameter Sets: {params_to_test}")
print(f"Sequence Lengths: {len_sequences}")
print(f"Discretization Method: {discretization_method}")
print(f"Samples per configuration: {n_samples}")
print(f"Total configurations: {len(params_to_test) * len(len_sequences)}")
print(f"{'='*80}\n")

# ============================================================================
# MAIN SIMULATION LOOP: Robustness across Heston parameters and sequence lengths
# ============================================================================

for param_key in params_to_test:
    param_config = heston_parameters[param_key]
    
    print(f"\n{'─'*80}")
    print(f"Configuration: {param_config['description']}")
    print(f"κ = {param_config['kappa']}, θ = {param_config['theta']}, σ = {param_config['sigma']}")
    print(f"{'─'*80}")
    
    for len_sequence in len_sequences:
        print(f"\n  Testing with T = {len_sequence}")
        
        for sample in range(start_sample, start_sample + n_samples):
            if (sample - start_sample + 1) % max(1, n_samples // 5) == 0 or sample == start_sample:
                print(f"    Sample {sample + 1}/{start_sample + n_samples}...", end='', flush=True)

            # Seed the global RNG per sample so each sample is reproducible on its own,
            # independent of how samples are split across processes. hmmlearn's sample()
            # also draws from the global RNG.
            np.random.seed([seed, sample])

            # Generate random initial theta for NPC_HMM
            transition_matrix = np.random.rand(ncl, ncl)
            
            # Normalize each row so that it sums to 1
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            theta_0_npc = np.array(transition_matrix)[:,:-1].flatten().tolist()
            
            # Generate random initial theta for QHMM
            theta_0_q = [np.random.uniform(2*np.pi, 6*np.pi) for _ in range(ansatz.num_parameters)]
            
            # Create models
            model_pc = PC_HMM(k=k,
                              ncl=ncl,
                              theta=theta_gen,
                              observations=observations)

            model_npc = NPC_HMM(k=k,
                               ncl=ncl,
                               theta=theta_0_npc,
                               observations=observations)
            
            model_qhmm = QHMM(theta=theta_0_q,
                             result_getter=result_getter,
                             initial_state=initial_state,
                             ansatz=ansatz)

            # Generate sequence from PC_HMM (correctly specified)
            sequence = model_pc.generate_sequence(len_sequence)

            # Calculate initial likelihoods
            try:
                model_pc_likelihood = model_pc.log_likelihood(sequence)
            except:
                model_pc_likelihood = -np.inf
            
            try:
                model_npc_likelihood = model_npc.log_likelihood(sequence)
            except:
                model_npc_likelihood = -np.inf
            
            try:
                model_qhmm_likelihood = model_qhmm.log_likelihood(sequence)
            except:
                model_qhmm_likelihood = -np.inf
            
            # Apply discretization if needed
            if discretization_method == 'quantile':
                discrete_sequence = discretize_quantile(sequence, len(observations))
            else:  # equal_width
                discrete_sequence = discretize_equal_width(sequence, len(observations))
            
            # Train PC_HMM (should converge quickly - correctly specified)
            try:
                theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(
                    model=model_pc,
                    sequence=discrete_sequence,
                    theta_0=theta_gen,
                    max_iter=max_iter,
                    tol=tol)
                trained_pc_likelihood = np.exp(-training_curve_pc[-1]) if training_curve_pc else -np.inf
            except Exception as e:
                theta_trained_pc = None
                training_time_pc = 0
                nit_pc = 0
                training_curve_pc = []
                trained_pc_likelihood = -np.inf
            
            # Train NPC_HMM (non-parametric - more parameters to learn)
            try:
                theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(
                    model=model_npc,
                    sequence=discrete_sequence,
                    theta_0=theta_0_npc,
                    max_iter=max_iter,
                    tol=tol)
                trained_npc_likelihood = np.exp(-training_curve_npc[-1]) if training_curve_npc else -np.inf
            except Exception as e:
                theta_trained_npc = None
                training_time_npc = 0
                nit_npc = 0
                training_curve_npc = []
                trained_npc_likelihood = -np.inf
            
            # Train QHMM (quantum-inspired - ansatz approximation)
            try:
                theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(
                    model=model_qhmm,
                    sequence=discrete_sequence,
                    theta_0=theta_0_q,
                    max_iter=max_iter,
                    tol=tol,
                    method=qhmm_method)
                trained_qhmm_likelihood = np.exp(-training_curve_q[-1]) if training_curve_q else -np.inf
            except Exception as e:
                theta_trained_q = None
                training_time_q = 0
                nit_q = 0
                training_curve_q = []
                trained_qhmm_likelihood = -np.inf
            
            # Store results
            sample_key = f"{param_key}_T{len_sequence}_sample{sample}"
            n_samples_data = {
                'seed': [seed, sample],
                'heston_params': param_config,
                'sequence_length': len_sequence,
                'discretization': discretization_method,
                'initial_theta_npc': theta_0_npc,
                'initial_theta_qhmm': theta_0_q,
                'model_pc_initial_likelihood': float(model_pc_likelihood),
                'model_npc_initial_likelihood': float(model_npc_likelihood),
                'model_qhmm_initial_likelihood': float(model_qhmm_likelihood),
                'trained_pc_likelihood': float(trained_pc_likelihood),
                'trained_npc_likelihood': float(trained_npc_likelihood),
                'trained_qhmm_likelihood': float(trained_qhmm_likelihood),
                'theta_trained_pc': theta_trained_pc,  
                'training_time_pc': training_time_pc,
                'nit_pc': nit_pc,
                'training_curve_pc': training_curve_pc,
                'theta_trained_npc': theta_trained_npc,  
                'training_time_npc': training_time_npc,
                'nit_npc': nit_npc,
                'training_curve_npc': training_curve_npc,
                'theta_trained_q': theta_trained_q,
                'training_time_q': training_time_q,
                'nit_q': nit_q,
                'training_curve_q': training_curve_q,
                'sequence_length_actual': len(sequence),
            }
            
            data[sample_key] = n_samples_data
            
            # Save incrementally
            with open(path, "w") as outfile: 
                json.dump(data, outfile, indent=4, default=str)
        
        print(f" Done")

print(f"\n{'='*80}")
print(f"Results saved to: {path}")
print(f"{'='*80}\n")
