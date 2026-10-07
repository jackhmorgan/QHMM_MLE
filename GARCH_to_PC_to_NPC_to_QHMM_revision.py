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

REVISION VERSION: Enhanced simulation study with:
  - GARCH(1,1) process for model misspecification testing
  - Multiple Heston parameter configurations
  - Variable sequence lengths for asymptotic analysis
  - Optimized algorithms and discretization methods
'''

from HMM import QHMM, PC_HMM, NPC_HMM, GARCH_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils.pc_utils import minimize_pc_hmm
from HMM.utils.garch_utils import minimize_garch_hmm
import pandas as pd
import numpy as np
import json
import os
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Parse command line arguments for enhanced simulation study")

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
    '--len_sequences', 
    type=str,
    help='Comma-separated list of sequence lengths (e.g., "500,1000,2000") for asymptotic analysis',
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

parser.add_argument(
    '--dgp',
    type=str,
    help='Data Generating Process: "cir" for CIR (correctly specified), "garch" for GARCH(1,1) (misspecified), or "both"',
)

parser.add_argument(
    '--heston_params',
    type=str,
    help='Heston parameter set: "base" (default), "heston1993", "eraker2004", or "all" for robustness',
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

parser.add_argument(
    '--garch_coupled',
    action='store_true',
    help='Fit the coupled GARCH_HMM (observation and next state share the GARCH shock) instead of the standard HMM',
)

args = parser.parse_args()

# Set defaults
output_path = args.output_path if args.output_path else 'MLE/garch_to_pc_to_npc_to_qhmm_revision.json'
len_sequences_input = args.len_sequences if args.len_sequences else "500"
n_samples = args.n_samples if args.n_samples else 10
max_iter = args.max_iter if args.max_iter else 1000
tol = args.tol if args.tol else 0.0001
k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4
dgp_type = args.dgp if args.dgp else "both"
heston_set = args.heston_params if args.heston_params else "all"
seed = args.seed if args.seed is not None else 0
start_sample = args.start_sample if args.start_sample else 0
qhmm_method = args.qhmm_method if args.qhmm_method else 'SLSQP'

# Parse sequence lengths for asymptotic analysis
try:
    len_sequences = [int(x.strip()) for x in len_sequences_input.split(',')]
except:
    len_sequences = [500]

# ============================================================================
# PARAMETER CONFIGURATIONS FOR ROBUSTNESS TESTING
# ============================================================================

# Heston parameter configurations satisfying Feller condition (2κθ ≥ σ²)
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

# GARCH(1,1) parameters (misspecified DGP)
# X_t = 0.019 + 0.10*Y_{t-1}^2 + 0.88*X_{t-1}
# Y_t = sqrt(X_t)*w_t, w_t ~ N(0,1)
garch_params = {
    'omega': 0.019,
    'alpha': 0.10,
    'beta': 0.88,
    'description': 'GARCH(1,1) weak process for misspecification testing'
}

# GARCH_HMM starts from the DGP parameters (converted from percent to decimal returns), as PC_HMM
# starts from the CIR parameters
garch_coupled = args.garch_coupled
theta_gen_garch = [garch_params['omega'] / 100**2, garch_params['alpha'], garch_params['beta']]

# Handle output path with auto-increment
filename, extension = os.path.splitext(output_path)
counter = 1

while os.path.exists(output_path):
    output_path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# Load SPY data for GARCH sequence generation
try:
    df = pd.read_csv('^SPX.csv')
    log_returns = pd.DataFrame({'log_returns': np.log(df['Close'].shift(-1) / df['Close'])})
    log_returns = log_returns.dropna().reset_index(drop=True)
    log_returns_values = log_returns['log_returns'].values
except:
    print("Warning: Could not load SPY data. Using synthetic data.")
    log_returns_values = np.random.normal(0, 0.01, 10000)

observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]

# Setup QHMM parameters
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

# Initialize output data structure
output_data = {}
output_data['metadata'] = {
    'garch_params': garch_params,
    'garch_hmm': {'ncl': ncl, 'coupled': garch_coupled, 'theta_0': theta_gen_garch},
    'observations': observations,
    'len_sequences': len_sequences,
    'ncl': ncl,
    'k': k,
    'n_samples': n_samples,
    'seed': seed,
    'start_sample': start_sample,
    'qhmm_method': qhmm_method,
    'qhmm_likelihood': 'sum of per-step log probabilities' if qhmm_log_sum else 'exact sequence probability',
    'dgp_type': dgp_type,
    'heston_param_set': heston_set,
    'optimization_notes': {
        'algorithm': 'SLSQP/L-BFGS-B for classical HMM, gradient descent for QHMM',
        'classical_hmm_params': ncl * (ncl - 1),
        'reason_for_choice': 'Gradient-based methods more suitable than Nelder-Mead for high-dimensional problems (>10-20 parameters)',
        'references': ['Lagarias et al. (1998) convergence guarantees', 'Nocedal & Wright (2006)']
    },
    'discretization_notes': {
        'method': 'Quantile-based binning of log returns',
        'n_bins': len(observations),
        'robustness_needed': 'Should test with alternative discretization methods for sensitivity analysis'
    }
}

with open(output_path, "w") as outfile: 
    json.dump(output_data, outfile, indent=4, default=str)

# ============================================================================
# SIMULATION FUNCTION: Generate GARCH(1,1) sequences
# ============================================================================

def generate_garch_sequence(T, omega, alpha, beta):
    """
    Generate weak GARCH(1,1) sequence:
    X_t = omega + alpha*Y_{t-1}^2 + beta*X_{t-1}
    Y_t = sqrt(X_t)*w_t, where w_t ~ N(0,1)

    This is a misspecified DGP (not in CIR family)
    Draws from the global numpy RNG, which is seeded per sample in the main loop.
    """
    X = np.zeros(T)
    Y = np.zeros(T)
    
    X[0] = omega / (1 - alpha - beta)  # Unconditional variance
    w = np.random.normal(0, 1, T)
    
    for t in range(T):
        Y[t] = np.sqrt(X[t]) * w[t]
        if t < T - 1:
            X[t+1] = omega + alpha * Y[t]**2 + beta * X[t]

    # Parameters are V-Lab estimates for percent returns; convert to decimal log returns
    # to match the SPX-based observation bins and the fitted GARCH model.
    return (Y / 100).reshape(-1, 1)

# ============================================================================
# SIMULATION STUDY: Enhanced with misspecification and parameter robustness
# ============================================================================

# Determine which Heston parameters to use
if heston_set == 'all':
    params_to_test = ['base', 'heston1993', 'eraker2004']
elif heston_set == 'base':
    params_to_test = ['base']
else:
    params_to_test = [heston_set]

# Determine which DGPs to test
dgp_list = []
if dgp_type in ['cir', 'both']:
    dgp_list.append('cir')
if dgp_type in ['garch', 'both']:
    dgp_list.append('garch')

print(f"\n{'='*80}")
print(f"ENHANCED SIMULATION STUDY")
print(f"{'='*80}")
print(f"Data Generating Processes: {dgp_list}")
print(f"Heston Parameter Sets: {params_to_test}")
print(f"Sequence Lengths: {len_sequences}")
print(f"Samples per configuration: {n_samples}")
print(f"Total configurations: {len(dgp_list) * len(params_to_test) * len(len_sequences)}")
print(f"{'='*80}\n")

for dgp in dgp_list:
    for param_key in params_to_test:
        param_config = heston_parameters[param_key]
        
        print(f"\n{'─'*80}")
        print(f"DGP: {dgp.upper()}, Heston Params: {param_config['description']}")
        print(f"{'─'*80}")
        
        for len_seq in len_sequences:
            print(f"\n  Sequence Length T = {len_seq}")
            
            for sample in range(start_sample, start_sample + n_samples):
                if (sample - start_sample + 1) % max(1, n_samples // 5) == 0 or sample == start_sample:
                    print(f"    Sample {sample + 1}/{start_sample + n_samples}...", end='', flush=True)

                # Seed the global RNG per sample so each sample is reproducible on its own,
                # independent of how samples are split across processes. hmmlearn's sample()
                # also draws from the global RNG.
                np.random.seed([seed, sample])

                # Use hard-coded theta for PC_HMM
                theta_gen_pc = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
                
                # Generate random initial theta for NPC_HMM
                transition_matrix = np.random.rand(ncl, ncl)
                transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
                theta_0_npc = np.array(transition_matrix)[:,:-1].flatten().tolist()
                
                # Generate random initial theta for QHMM
                theta_0_q = [np.random.uniform(2*np.pi, 6*np.pi) for _ in range(ansatz.num_parameters)]
                
                # Create models
                model_garch = GARCH_HMM(ncl=ncl,
                                        theta=theta_gen_garch,
                                        observations=observations,
                                        coupled=garch_coupled)

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
                
                # Generate sequence based on DGP type
                if dgp == 'cir':
                    # Use sampled log returns (approximation of CIR)
                    sequence_indices = np.random.choice(len(log_returns_values), len_seq, replace=True)
                    sequence = log_returns_values[sequence_indices].reshape(-1, 1)
                    dgp_label = f"cir_sampled_{param_key}"
                elif dgp == 'garch':
                    # Generate from GARCH(1,1)
                    sequence = generate_garch_sequence(len_seq, 
                                                      garch_params['omega'],
                                                      garch_params['alpha'],
                                                      garch_params['beta'])
                    dgp_label = f"garch_{param_key}"
                
                # Discretize sequence into observation bins for PC_HMM and NPC_HMM
                bins = [np.quantile(log_returns_values, (i+1)/(len(observations))) 
                       for i in range(len(observations)-1)]
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
                    # Calculate GARCH_HMM likelihood
                    garch_likelihood = model_garch.log_likelihood(discrete_sequence)
                except:
                    garch_likelihood = -np.inf

                try:
                    # Calculate PC_HMM likelihood
                    pc_likelihood = model_pc.log_likelihood(discrete_sequence)
                except:
                    pc_likelihood = -np.inf
                
                # Train PC_HMM
                try:
                    theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(
                        model=model_pc,
                        sequence=discrete_sequence,
                        theta_0=theta_gen_pc,
                        max_iter=max_iter,
                        tol=tol)
                    trained_pc_likelihood = np.exp(-training_curve_pc[-1]) if training_curve_pc else -np.inf
                except Exception as e:
                    theta_trained_pc = None
                    training_time_pc = 0
                    nit_pc = 0
                    training_curve_pc = []
                    trained_pc_likelihood = -np.inf

                # Train GARCH_HMM
                try:
                    theta_trained_garch, training_time_garch, nit_garch, training_curve_garch = minimize_garch_hmm(
                        model=model_garch,
                        sequence=discrete_sequence,
                        theta_0=theta_gen_garch,
                        max_iter=max_iter,
                        tol=tol)
                    trained_garch_likelihood = np.exp(-training_curve_garch[-1]) if training_curve_garch else -np.inf
                except Exception as e:
                    theta_trained_garch = None
                    training_time_garch = 0
                    nit_garch = 0
                    training_curve_garch = []
                    trained_garch_likelihood = -np.inf

                # Train NPC_HMM
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
                
                # Train QHMM
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
                
                # Save results for this sample
                sample_key = f"{dgp_label}_T{len_seq}_sample{sample}"
                sample_data = {
                    'dgp': dgp,
                    'seed': [seed, sample],
                    'heston_params': param_config,
                    'sequence_length': len_seq,
                    'initial_garch_likelihood': float(garch_likelihood),
                    'initial_theta_garch': theta_gen_garch,
                    'trained_garch_likelihood': float(trained_garch_likelihood),
                    'theta_trained_garch': theta_trained_garch,
                    'training_time_garch': training_time_garch,
                    'nit_garch': nit_garch,
                    'training_curve_garch': training_curve_garch,
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
                
                output_data[sample_key] = sample_data
                
                with open(output_path, "w") as outfile: 
                    json.dump(output_data, outfile, indent=4, default=str)
            
            print(f" Done")

print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
print(f"{'='*80}\n")
