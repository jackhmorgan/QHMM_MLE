import pandas as pd
import numpy as np
from HMM import PC_HMM
from HMM.utils.pc_utils import minimize_pc_hmm
import argparse
import numpy.random as random
import json
import os

parser = argparse.ArgumentParser(description="Parse command line arguments")

# Argument 1: List of numbers
parser.add_argument(
    '--n_samples', 
    type=int,  # Convert each input to a float
    help='Number of repetitions',
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

path = args.path if args.path else 'MLE/ClassicalConvergence/spy_pc_test.json'
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

df = pd.read_csv('MLE/^SPX.csv')

log_returns = pd.DataFrame({'log_returns': np.log(df['Close'].shift(-1) / df['Close'])})

log_returns = log_returns.dropna().reset_index(drop=True)

description = log_returns['log_returns'].describe()

num_iterations = 5
no = 4
bins = [log_returns['log_returns'].quantile((i+1)/(no)) for i in range(no-1)]
observations = [log_returns['log_returns'].quantile((i+1)/(no+1)) for i in range(no)]


sequence = []

for lr in df['log_returns']:
    sorted = False
    for i, edge in enumerate(bins):
        if lr > edge and sorted==False:
            sequence.append(i)
            sorted == True
    if sorted==False:
        sequence.append(len(bins))

sequence = np.array(sequence).reshape(-1,1)

model = PC_HMM(k=k,
                ncl=ncl,
                observations=observations)

data = {}
data['ncl'] = ncl
data['k'] = k
data['max_iter'] = max_iter
data['tol'] = tol

for iteration in range(n_samples):

    done = False
    while done==False:
        # Generate random values for a, b, and c within the specified range
        alpha = random.uniform(1, 10)
        beta = random.uniform(0.01, 0.1)
        sigma = random.uniform(0.1, 1.0)
        
        # Check if the condition 4ab > c^2 is satisfied
        if 4 * alpha * beta > sigma**2:
            theta_0 = [alpha, beta, sigma]
            done = True

    trained_theta, training_time, nit, training_curve = minimize_pc_hmm(model=model,
                sequence=sequence,
                theta_0=theta_0,
                max_iter=max_iter,
                tol=1e-6)
    
    data[iteration] = {'theta_0' : theta_0,
                       'trained_theta' : trained_theta,
                       'training_time' : training_time,
                       'nit' : nit,
                       'final_likelihood' : training_curve[-1],
                       'training_curve' : training_curve}
    
    with open(path, "w") as outfile: 
        json.dump(data, outfile, indent=4)