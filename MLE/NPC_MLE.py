from hmmlearn import hmm
from scipy.optimize import minimize
import numpy as np
import sys
import json
import os
import argparse
import time

from HMM import NPC_HMM
from HMM.utils import calculate_mse

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
path = args.path if args.path else 'MLE/ClassicalConvergence/npc_test.json'
sample_length = args.sample_length if args.sample_length else 1600 # 16 parameters x 100 
n_samples_list = args.n_samples_list if args.n_samples_list else [10, 100, 1000]
max_iter = args.max_iter if args.max_iter else 100
tol = args.tol if args.tol else 0.01

filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

ncl = 4
k=4
observations = [-1, -0.33, 0.33, 1]
# Define the initial latent state and emission matrix
initial_latent_state = [0.4, 0.3, 0.1, 0.2]
emission_matrix = [[0.1, 0.3, 0.5, 0.1],
                   [0.6, 0.1, 0.1, 0.2],
                   [0.2, 0.2, 0.2, 0.4],
                   [0.1, 0.7, 0.1, 0.1]]

# Define the ideam transition matrix, and the starting transition matrix that we will try to train into the ideal
transition_matrix_1 = [[0.4, 0.3, 0.2, 0.1],
                       [0.1, 0.7, 0.1, 0.1],
                       [0.15, 0.15, 0.4, 0.3],
                       [0.3, 0.15, 0.15, 0.4]]

transition_matrix_2 = [[0.35766901, 0.19469946, 0.33753688, 0.11009465],
                       [0.1457671,  0.5914749,  0.175172,   0.087586],
                       [0.29928881, 0.11701788, 0.31136717, 0.27232615],
                       [0.21261512, 0.1400294, 0.11754035, 0.52981513]]

test_tm = [[0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25],]

# transition_matrix_2 = [[0.39972015, 0.29973213, 0.20209149, 0.09845622],
#                        [0.10051336, 0.70110495, 0.09886772, 0.09951397],
#                        [0.14935292, 0.14969928, 0.4010547 , 0.2998931 ],
#                        [0.2985524 , 0.1504031 , 0.15058077, 0.40046373]]
    
data = {}
likelihood_curve = []

def theta_to_matrix(theta):
    matrix = np.zeros((ncl, ncl), dtype=np.float64)
    index = 0
    for row in range(ncl):
        for column in range(ncl-1):
            matrix[row][column] = theta[index]
            index += 1
        matrix[row][-1] = 1-np.sum(matrix[row])
    return matrix

def matrix_to_theta(matrix):
    theta = np.array(matrix)[:,:-1].flatten().tolist()
    return theta

theta_gen = matrix_to_theta(transition_matrix_1)
#theta_0 = matrix_to_theta(transition_matrix_2)
theta_0 = matrix_to_theta(test_tm)

data['gen_theta'] = theta_gen
data['start_theta'] = theta_0
data['start_mse'] = calculate_mse(theta=theta_0, theta_true=theta_gen)
        
for n_samples in n_samples_list:
    lengths = [sample_length for _ in range(n_samples)]
    
    model_1 = NPC_HMM(k=k,
                      ncl=ncl,
                      theta = theta_gen,
                      observations = observations,
                      mean=0.127)

    model_2 = model_2 = NPC_HMM(k=k,
                      ncl=ncl,
                      theta=theta_0,
                      observations = observations,
                      mean=0.127)

    sequence = model_1.generate_sequence(sample_length*n_samples)

    def neg_log_likelihood(theta):
        print('---------')
        print(theta)
        print(type(theta))
        penalty = 0
        likelihood = 0
        penalizer = sample_length**4
        for param in theta:
            if param < 0:
                penalty += (1-param)*penalizer
                print('penalty, param < 0 : ', param)
            
            if param > 1:
                penalty += param*penalizer
                print('penalty, param > 1: ', param)
    
        for i in range(0, ncl*(ncl-1), ncl-1):
            sum_col = np.sum(theta[i:i + ncl-1])
            if sum_col > 1:
                penalty += sum_col*penalizer
                print('penalty, column: ', theta[i:i + ncl-1])
        if penalty == 0:
            model_2.update_theta(theta)
            likelihood = model_2.log_likelihood(sequence)
        
        print('penalty: ', penalty)
        print('likelihood: ', -likelihood)
        if not likelihood == 0:
            likelihood_curve.append(likelihood)
        return -likelihood + penalty

    gen_model_likelihood = model_1.log_likelihood(sequence)
    init_model_likelihood = model_2.log_likelihood(sequence)
    
    
    start_time = time.time()
    likelihood_curve = []
    # Optimize
    result = minimize(neg_log_likelihood, 
                      theta_0, 
                      method='Nelder-Mead',
                      options = {'maxiter': 100},
                      )
    training_time = time.time() - start_time
    
    trained_model_likelihood = model_2.log_likelihood(sequence)
    
    trained_theta = result.x.tolist()
    trained_mse = calculate_mse(theta=trained_theta, theta_true=theta_gen)
    
    n_samples_data = {'training_time' : training_time,
                      'gen_model_likelihood' : gen_model_likelihood,
                      'init_model_likelihood' : init_model_likelihood,
                      'trained_model_likelihood' : trained_model_likelihood,
                      'trained_mse': trained_mse,
                      'trained_theta' : trained_theta,
                      'likelihood_curve': likelihood_curve}
    data[n_samples] = n_samples_data
    
    with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)
    

