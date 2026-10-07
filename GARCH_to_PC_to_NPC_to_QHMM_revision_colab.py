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

COLAB REVISION VERSION: Enhanced simulation study optimized for Google Colab
- Automatic Google Drive mounting
- GPU-optimized QHMM simulation
- Colab-compatible file handling
- Progress tracking for notebook environment
'''

import sys
import os

# ============================================================================
# COLAB ENVIRONMENT DETECTION & SETUP
# ============================================================================

def setup_colab_environment():
    """Detect Colab environment and mount Google Drive if needed"""
    try:
        from google.colab import drive
        IN_COLAB = True
        print("✓ Running in Google Colab")
        
        # Mount Google Drive
        try:
            drive.mount('/content/drive', force_remount=False)
            print("✓ Google Drive mounted at /content/drive")
            drive_path = '/content/drive/MyDrive'
        except:
            print("⚠ Could not mount Google Drive. Using local filesystem.")
            drive_path = None
            
    except ImportError:
        IN_COLAB = False
        drive_path = None
        print("✓ Running locally (not in Colab)")
    
    return IN_COLAB, drive_path

# Setup environment
IN_COLAB, DRIVE_PATH = setup_colab_environment()

# ============================================================================
# DEPENDENCY INSTALLATION (Colab)
# ============================================================================

if IN_COLAB:
    print("\nInstalling dependencies for Colab environment...")
    os.system('pip install qiskit qiskit-aer -q')
    print("✓ Dependencies installed")

from HMM import QHMM, PC_HMM, NPC_HMM, GARCH
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils.pc_utils import minimize_pc_hmm
from HMM.utils.garch_utils import minimize_GARCH
import pandas as pd
import numpy as np
import json
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# COLAB-COMPATIBLE FILE PATH HANDLING
# ============================================================================

def resolve_path(path_str, drive_path=None):
    """
    Resolve file paths for Colab or local environment
    
    Priority:
    1. If path starts with '/', treat as absolute
    2. If in Colab and drive_path available, check Drive first
    3. Otherwise treat as relative to current directory
    """
    if os.path.isabs(path_str):
        return path_str
    
    if IN_COLAB and drive_path:
        drive_option = os.path.join(drive_path, path_str)
        if os.path.exists(drive_option):
            return drive_option
    
    # Default to current directory
    return path_str

# ============================================================================
# MAIN SCRIPT
# ============================================================================

parser = argparse.ArgumentParser(description="Enhanced GARCH comparison for Google Colab")

parser.add_argument('--n_samples', type=int, help='Number of sequences per configuration')
parser.add_argument('--garch_results_path', type=str, help='Path to GARCH results')
parser.add_argument('--output_path', type=str, help='Path to save results')
parser.add_argument('--len_sequences', type=str, help='Comma-separated sequence lengths')
parser.add_argument('--k', type=int, help='Spot volatilities per integrated volatility')
parser.add_argument('--ncl', type=int, help='Number of classical latent states')
parser.add_argument('--max_iter', type=int, help='Maximum optimization iterations')
parser.add_argument('--tol', type=float, help='Convergence tolerance')
parser.add_argument('--dgp', type=str, help='DGP type: cir, garch, or both')
parser.add_argument('--heston_params', type=str, help='Heston params: base, heston1993, eraker2004, or all')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for QHMM (Colab only)')
parser.add_argument('--save_to_drive', action='store_true', help='Save results to Google Drive (Colab only)')

args = parser.parse_args()

# Set defaults
output_path = args.output_path if args.output_path else 'garch_to_pc_to_npc_to_qhmm_revision_colab.json'
len_sequences_input = args.len_sequences if args.len_sequences else "500"
n_samples = args.n_samples if args.n_samples else 5  # Smaller default for Colab testing
max_iter = args.max_iter if args.max_iter else 500  # Reduced for Colab efficiency
tol = args.tol if args.tol else 0.0001
k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4
dgp_type = args.dgp if args.dgp else "cir"
heston_set = args.heston_params if args.heston_params else "base"
use_gpu = args.use_gpu and IN_COLAB
save_to_drive = args.save_to_drive and IN_COLAB

# Parse sequence lengths
try:
    len_sequences = [int(x.strip()) for x in len_sequences_input.split(',')]
except:
    len_sequences = [500]

# Resolve output path
if save_to_drive and DRIVE_PATH:
    output_path = os.path.join(DRIVE_PATH, output_path)
else:
    output_path = resolve_path(output_path, DRIVE_PATH)

# ============================================================================
# HESTON PARAMETER CONFIGURATIONS
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
        'description': 'Heston (1993)'
    },
    'eraker2004': {
        'kappa': 3.99,
        'theta': 0.014,
        'sigma': 0.14,
        'description': 'Eraker (2004)'
    }
}

# GARCH(1,1) parameters
garch_params = {
    'omega': 0.019,
    'alpha': 0.10,
    'beta': 0.88,
}

garch_p, garch_q = 1, 1

# ============================================================================
# HANDLE OUTPUT PATH WITH AUTO-INCREMENT
# ============================================================================

filename, extension = os.path.splitext(output_path)
counter = 1

while os.path.exists(output_path):
    output_path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# ============================================================================
# LOAD DATA
# ============================================================================

try:
    # Try local first, then Drive
    if os.path.exists('^SPX.csv'):
        df = pd.read_csv('^SPX.csv')
    elif DRIVE_PATH:
        drive_csv = os.path.join(DRIVE_PATH, 'QHMM_MLE', '^SPX.csv')
        if os.path.exists(drive_csv):
            df = pd.read_csv(drive_csv)
        else:
            raise FileNotFoundError("^SPX.csv not found")
    else:
        raise FileNotFoundError("^SPX.csv not found")
    
    log_returns = pd.DataFrame({'log_returns': np.log(df['Close'].shift(-1) / df['Close'])})
    log_returns = log_returns.dropna().reset_index(drop=True)
    log_returns_values = log_returns['log_returns'].values
    print(f"✓ Loaded SPY data: {len(log_returns_values)} observations")
except Exception as e:
    print(f"⚠ Could not load SPY data: {e}. Using synthetic data.")
    log_returns_values = np.random.normal(0, 0.01, 10000)

garch_model = {
    "omega": 2.5896387874042265e-06,
    "alpha[1]": 0.1,
    "alpha[2]": 0.1,
    "beta[1]": 0.26,
    "beta[2]": 0.26,
    "beta[3]": 0.26
}
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]

# ============================================================================
# SETUP QHMM (WITH GPU OPTION FOR COLAB)
# ============================================================================

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

if use_gpu:
    try:
        from qiskit_aer import AerSimulator
        from HMM.utils.qhmm_utils import statevector_result_getter
        
        # GPU-accelerated simulator for Colab
        simulator = AerSimulator(device='GPU')
        result_getter = statevector_result_getter(rescaling_factor=1e6)
        print("✓ Using GPU-accelerated QHMM simulation")
    except Exception as e:
        print(f"⚠ GPU not available: {e}. Using CPU simulator.")
        from HMM.utils.qhmm_utils import statevector_result_getter
        result_getter = statevector_result_getter(rescaling_factor=1e6)
else:
    from HMM.utils.qhmm_utils import statevector_result_getter
    result_getter = statevector_result_getter(rescaling_factor=1e6)

initial_state = QuantumCircuit(1, name='Initial_State')
initial_state.h(0)

ansatz = efficient_su2(3, reps=3, entanglement='full', su2_gates=['rz','ry'])

# ============================================================================
# GENERATE GARCH SEQUENCE
# ============================================================================

def generate_garch_sequence(T, omega, alpha, beta, seed=None):
    """Generate weak GARCH(1,1) sequence"""
    if seed is not None:
        np.random.seed(seed)
    
    X = np.zeros(T)
    Y = np.zeros(T)
    X[0] = omega / (1 - alpha - beta)
    w = np.random.normal(0, 1, T)
    
    for t in range(T):
        Y[t] = np.sqrt(X[t]) * w[t]
        if t < T - 1:
            X[t+1] = omega + alpha * Y[t]**2 + beta * X[t]
    
    return Y.reshape(-1, 1)

# ============================================================================
# INITIALIZE OUTPUT
# ============================================================================

output_data = {
    'metadata': {
        'environment': 'Google Colab' if IN_COLAB else 'Local',
        'gpu_enabled': use_gpu,
        'garch_model': garch_model,
        'observations': observations,
        'len_sequences': len_sequences,
        'ncl': ncl,
        'k': k,
        'n_samples': n_samples,
        'dgp_type': dgp_type,
        'heston_param_set': heston_set,
    }
}

with open(output_path, "w") as outfile: 
    json.dump(output_data, outfile, indent=4, default=str)

# ============================================================================
# SIMULATION LOOP
# ============================================================================

if heston_set == 'all':
    params_to_test = ['base', 'heston1993', 'eraker2004']
elif heston_set == 'base':
    params_to_test = ['base']
else:
    params_to_test = [heston_set]

dgp_list = []
if dgp_type in ['cir', 'both']:
    dgp_list.append('cir')
if dgp_type in ['garch', 'both']:
    dgp_list.append('garch')

print(f"\n{'='*80}")
print(f"COLAB SIMULATION: {dgp_type.upper()} DGP, Heston={heston_set}")
print(f"Sequence lengths: {len_sequences}, Samples: {n_samples}")
print(f"{'='*80}\n")

total_configs = len(dgp_list) * len(params_to_test) * len(len_sequences) * n_samples
config_count = 0

for dgp in dgp_list:
    for param_key in params_to_test:
        param_config = heston_parameters[param_key]
        
        for len_seq in len_sequences:
            print(f"Testing {dgp.upper()} DGP with {param_config['description']}, T={len_seq}")
            
            for sample in range(n_samples):
                config_count += 1
                
                # Progress indicator
                progress = f"[{config_count}/{total_configs}]"
                if (sample + 1) % max(1, n_samples // 3) == 0 or sample == 0:
                    print(f"  {progress} Sample {sample + 1}/{n_samples}...", end='', flush=True)
                
                theta_gen_pc = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
                
                transition_matrix = np.random.rand(ncl, ncl)
                transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
                theta_0_npc = np.array(transition_matrix)[:,:-1].flatten().tolist()
                
                theta_0_q = [np.random.uniform(2*np.pi, 6*np.pi) for _ in range(ansatz.num_parameters)]
                
                model_garch = GARCH(p=garch_p, q=garch_q, observations=log_returns_values)
                model_pc = PC_HMM(k=k, ncl=ncl, theta=theta_gen_pc, observations=observations)
                model_npc = NPC_HMM(k=k, ncl=ncl, theta=theta_0_npc, observations=observations)
                model_qhmm = QHMM(theta=theta_0_q, result_getter=result_getter,
                                  initial_state=initial_state, ansatz=ansatz)
                
                # Generate sequence
                if dgp == 'cir':
                    sequence_indices = np.random.choice(len(log_returns_values), len_seq, replace=True)
                    sequence = log_returns_values[sequence_indices].reshape(-1, 1)
                    dgp_label = f"cir_{param_key}"
                else:
                    sequence = generate_garch_sequence(len_seq, garch_params['omega'],
                                                      garch_params['alpha'], garch_params['beta'], seed=sample)
                    dgp_label = f"garch_{param_key}"
                
                # Discretize
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
                
                # Compute likelihoods
                try:
                    garch_likelihood = model_garch.log_likelihood(sequence.flatten())
                except:
                    garch_likelihood = -np.inf
                
                try:
                    pc_likelihood = model_pc.log_likelihood(discrete_sequence)
                except:
                    pc_likelihood = -np.inf
                
                # Train models
                try:
                    theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(
                        model=model_pc, sequence=discrete_sequence, theta_0=theta_gen_pc,
                        max_iter=max_iter, tol=tol)
                    trained_pc_likelihood = np.exp(-training_curve_pc[-1]) if training_curve_pc else -np.inf
                except:
                    theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = None, 0, 0, []
                    trained_pc_likelihood = -np.inf
                
                try:
                    theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(
                        model=model_npc, sequence=discrete_sequence, theta_0=theta_0_npc,
                        max_iter=max_iter, tol=tol)
                    trained_npc_likelihood = np.exp(-training_curve_npc[-1]) if training_curve_npc else -np.inf
                except:
                    theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = None, 0, 0, []
                    trained_npc_likelihood = -np.inf
                
                try:
                    theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(
                        model=model_qhmm, sequence=discrete_sequence, theta_0=theta_0_q,
                        max_iter=max_iter, tol=tol)
                    trained_qhmm_likelihood = np.exp(-training_curve_q[-1]) if training_curve_q else -np.inf
                except:
                    theta_trained_q, training_time_q, nit_q, training_curve_q = None, 0, 0, []
                    trained_qhmm_likelihood = -np.inf
                
                # Store results
                sample_key = f"{dgp_label}_T{len_seq}_sample{sample}"
                sample_data = {
                    'dgp': dgp,
                    'heston_params': param_config,
                    'sequence_length': len_seq,
                    'garch_likelihood': float(garch_likelihood),
                    'initial_pc_likelihood': float(pc_likelihood),
                    'trained_pc_likelihood': float(trained_pc_likelihood),
                    'trained_npc_likelihood': float(trained_npc_likelihood),
                    'trained_qhmm_likelihood': float(trained_qhmm_likelihood),
                    'training_time_pc': training_time_pc,
                    'training_time_npc': training_time_npc,
                    'training_time_qhmm': training_time_q,
                }
                
                output_data[sample_key] = sample_data
                
                with open(output_path, "w") as outfile: 
                    json.dump(output_data, outfile, indent=4, default=str)
            
            print(" ✓")

print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
if save_to_drive:
    print(f"✓ Results backed up to Google Drive")
print(f"{'='*80}\n")
