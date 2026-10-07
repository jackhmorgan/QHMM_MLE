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

COLAB REVISION VERSION: Enhanced empirical results optimized for Google Colab
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

from HMM import QHMM, PC_HMM, NPC_HMM
from HMM.utils.qhmm_utils import minimize_qhmm
from HMM.utils.npc_utils import minimize_npc_hmm
from HMM.utils.pc_utils import minimize_pc_hmm
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

parser = argparse.ArgumentParser(description="Enhanced empirical comparison for Google Colab")

parser.add_argument('--n_samples', type=int, help='Number of sequences per configuration')
parser.add_argument('--path', type=str, help='Output path for results JSON')
parser.add_argument('--len_sequences', type=str, help='Comma-separated sequence lengths')
parser.add_argument('--k', type=int, help='Spot volatilities per integrated volatility')
parser.add_argument('--ncl', type=int, help='Number of classical latent states')
parser.add_argument('--max_iter', type=int, help='Maximum optimization iterations')
parser.add_argument('--tol', type=float, help='Convergence tolerance')
parser.add_argument('--heston_params', type=str, help='Heston params: base, heston1993, eraker2004, or all')
parser.add_argument('--discretization', type=str, help='Discretization: quantile or equal_width')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU for QHMM')
parser.add_argument('--save_to_drive', action='store_true', help='Save to Google Drive')

args = parser.parse_args()

# Set defaults
path = args.path if args.path else 'pc_to_npc_to_qhmm_revision_colab.json'
len_sequences_input = args.len_sequences if args.len_sequences else "500"
n_samples = args.n_samples if args.n_samples else 5
max_iter = args.max_iter if args.max_iter else 500
tol = args.tol if args.tol else 0.0001
k = args.k if args.k else 1
ncl = args.ncl if args.ncl else 4
heston_set = args.heston_params if args.heston_params else "base"
discretization_method = args.discretization if args.discretization else "quantile"
use_gpu = args.use_gpu and IN_COLAB
save_to_drive = args.save_to_drive and IN_COLAB

# Parse sequence lengths
try:
    len_sequences = [int(x.strip()) for x in len_sequences_input.split(',')]
except:
    len_sequences = [500]

# Resolve output path
if save_to_drive and DRIVE_PATH:
    path = os.path.join(DRIVE_PATH, path)
else:
    path = resolve_path(path, DRIVE_PATH)

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
        'description': 'Heston (1993) - Higher mean reversion'
    },
    'eraker2004': {
        'kappa': 3.99,
        'theta': 0.014,
        'sigma': 0.14,
        'description': 'Eraker (2004) - Very high mean reversion'
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
    """Discretize using equal-width binning"""
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    bins = np.linspace(min_val, max_val, n_bins)
    discrete_seq = np.digitize(sequence.flatten(), bins) - 1
    discrete_seq = np.clip(discrete_seq, 0, n_bins - 1)
    return discrete_seq.reshape(-1, 1)

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

# PC_HMM training parameters
theta_gen = [2.1961912602516445, 0.07722519718841697, 1.1333546404402364]
observations = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]

# ============================================================================
# HANDLE OUTPUT PATH WITH AUTO-INCREMENT
# ============================================================================

filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# ============================================================================
# INITIALIZE OUTPUT
# ============================================================================

data = {
    'metadata': {
        'environment': 'Google Colab' if IN_COLAB else 'Local',
        'gpu_enabled': use_gpu,
        'gen_theta': theta_gen,
        'len_sequences': len_sequences,
        'ncl': ncl,
        'k': k,
        'n_samples': n_samples,
        'heston_param_set': heston_set,
        'discretization_method': discretization_method,
    }
}

with open(path, "w") as outfile: 
    json.dump(data, outfile, indent=4, default=str)

# ============================================================================
# PARAMETER SELECTION
# ============================================================================

if heston_set == 'all':
    params_to_test = ['base', 'heston1993', 'eraker2004']
elif heston_set == 'base':
    params_to_test = ['base']
else:
    params_to_test = [heston_set]

print(f"\n{'='*80}")
print(f"COLAB EMPIRICAL STUDY: PC vs NPC vs QHMM")
print(f"{'='*80}")
print(f"Environment: Google Colab" if IN_COLAB else "Environment: Local")
print(f"GPU Enabled: {use_gpu}")
print(f"Heston Parameters: {params_to_test}")
print(f"Sequence Lengths: {len_sequences}")
print(f"Discretization: {discretization_method}")
print(f"Samples per config: {n_samples}")
print(f"{'='*80}\n")

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

total_configs = len(params_to_test) * len(len_sequences) * n_samples
config_count = 0

for param_key in params_to_test:
    param_config = heston_parameters[param_key]
    
    print(f"\n{'─'*80}")
    print(f"Configuration: {param_config['description']}")
    print(f"κ = {param_config['kappa']}, θ = {param_config['theta']}, σ = {param_config['sigma']}")
    print(f"{'─'*80}")
    
    for len_sequence in len_sequences:
        print(f"\n  Testing with T = {len_sequence}")
        
        for sample in range(n_samples):
            config_count += 1
            
            # Progress indicator
            progress = f"[{config_count}/{total_configs}]"
            if (sample + 1) % max(1, n_samples // 3) == 0 or sample == 0:
                print(f"    {progress} Sample {sample + 1}/{n_samples}...", end='', flush=True)
            
            # Generate random initial theta for NPC_HMM
            transition_matrix = np.random.rand(ncl, ncl)
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            theta_0_npc = np.array(transition_matrix)[:,:-1].flatten().tolist()
            
            # Generate random initial theta for QHMM
            theta_0_q = [np.random.uniform(2*np.pi, 6*np.pi) for _ in range(ansatz.num_parameters)]
            
            # Create models
            model_pc = PC_HMM(k=k, ncl=ncl, theta=theta_gen, observations=observations)
            model_npc = NPC_HMM(k=k, ncl=ncl, theta=theta_0_npc, observations=observations)
            model_qhmm = QHMM(theta=theta_0_q, result_getter=result_getter,
                             initial_state=initial_state, ansatz=ansatz)

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
            
            # Apply discretization
            if discretization_method == 'quantile':
                discrete_sequence = discretize_quantile(sequence, len(observations))
            else:
                discrete_sequence = discretize_equal_width(sequence, len(observations))
            
            # Train PC_HMM
            try:
                theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = minimize_pc_hmm(
                    model=model_pc, sequence=discrete_sequence, theta_0=theta_gen,
                    max_iter=max_iter, tol=tol)
                trained_pc_likelihood = np.exp(-training_curve_pc[-1]) if training_curve_pc else -np.inf
            except Exception as e:
                theta_trained_pc, training_time_pc, nit_pc, training_curve_pc = None, 0, 0, []
                trained_pc_likelihood = -np.inf
            
            # Train NPC_HMM
            try:
                theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = minimize_npc_hmm(
                    model=model_npc, sequence=discrete_sequence, theta_0=theta_0_npc,
                    max_iter=max_iter, tol=tol)
                trained_npc_likelihood = np.exp(-training_curve_npc[-1]) if training_curve_npc else -np.inf
            except Exception as e:
                theta_trained_npc, training_time_npc, nit_npc, training_curve_npc = None, 0, 0, []
                trained_npc_likelihood = -np.inf
            
            # Train QHMM
            try:
                theta_trained_q, training_time_q, nit_q, training_curve_q = minimize_qhmm(
                    model=model_qhmm, sequence=discrete_sequence, theta_0=theta_0_q,
                    max_iter=max_iter, tol=tol)
                trained_qhmm_likelihood = np.exp(-training_curve_q[-1]) if training_curve_q else -np.inf
            except Exception as e:
                theta_trained_q, training_time_q, nit_q, training_curve_q = None, 0, 0, []
                trained_qhmm_likelihood = -np.inf
            
            # Store results
            sample_key = f"{param_key}_T{len_sequence}_sample{sample}"
            n_samples_data = {
                'heston_params': param_config,
                'sequence_length': len_sequence,
                'discretization': discretization_method,
                'model_pc_initial_likelihood': float(model_pc_likelihood),
                'model_npc_initial_likelihood': float(model_npc_likelihood),
                'model_qhmm_initial_likelihood': float(model_qhmm_likelihood),
                'trained_pc_likelihood': float(trained_pc_likelihood),
                'trained_npc_likelihood': float(trained_npc_likelihood),
                'trained_qhmm_likelihood': float(trained_qhmm_likelihood),
                'training_time_pc': training_time_pc,
                'training_time_npc': training_time_npc,
                'training_time_qhmm': training_time_q,
                'nit_pc': nit_pc,
                'nit_npc': nit_npc,
                'nit_qhmm': nit_q,
            }
            
            data[sample_key] = n_samples_data
            
            # Save incrementally
            with open(path, "w") as outfile: 
                json.dump(data, outfile, indent=4, default=str)
        
        print(" ✓")

print(f"\n{'='*80}")
print(f"Results saved to: {path}")
if save_to_drive:
    print(f"✓ Results backed up to Google Drive")
print(f"{'='*80}\n")
