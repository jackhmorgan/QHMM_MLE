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

import pandas as pd
import numpy as np
from HMM import GARCH
import argparse
import json
import os
import time

parser = argparse.ArgumentParser(description="Parse command line arguments")

parser.add_argument(
    '--n_samples', 
    type=int,
    help='Number of different (p, q) combinations to try',
)

parser.add_argument(
    '--path', 
    type=str,
    help='Path to save results JSON file',
)

parser.add_argument(
    '--max_p',
    type=int,
    help='Maximum p (AR order) to test',
)

parser.add_argument(
    '--max_q',
    type=int,
    help='Maximum q (MA order) to test',
)

parser.add_argument(
    '--max_iter',
    type=int,
    help='The maximum number of optimization iterations',
)

parser.add_argument(
    '--mean',
    type=str,
    help='Mean model specification (Zero, Constant, AR, ARX)',
)

args = parser.parse_args()

path = args.path if args.path else 'spy_garch_results.json'
max_p = args.max_p if args.max_p else 3
max_q = args.max_q if args.max_q else 3
n_samples = args.n_samples if args.n_samples else 9
max_iter = args.max_iter if args.max_iter else 200
mean_model = args.mean if args.mean else 'Zero'

filename, extension = os.path.splitext(path)
counter = 1

while os.path.exists(path):
    path = filename + " (" + str(counter) + ")" + extension
    counter += 1

# Load data
df = pd.read_csv('^SPX.csv')

log_returns = pd.DataFrame({'log_returns': np.log(df['Close'].shift(-1) / df['Close'])})
log_returns = log_returns.dropna().reset_index(drop=True)

description = log_returns['log_returns'].describe()

# Initialize results dictionary
data = {}
data['max_p'] = max_p
data['max_q'] = max_q
data['max_iter'] = max_iter
data['mean_model'] = mean_model
data['n_observations'] = len(log_returns)
data['log_returns_describe'] = description.to_dict()

with open(path, "w") as outfile: 
    json.dump(data, outfile, indent=4)

# Train GARCH models with different (p, q) combinations
iteration = 0
for p in range(1, max_p + 1):
    for q in range(1, max_q + 1):
        if iteration >= n_samples:
            break
        
        data[iteration] = {}
        
        try:
            # Create and fit GARCH model
            model = GARCH(p=p, 
                         q=q, 
                         mean=mean_model,
                         observations=log_returns['log_returns'].values)
            
            start_time = time.time()
            fit_results = model.fit(disp='off', 
                                   max_iter=max_iter,
                                   show_warning=False)
            training_time = time.time() - start_time
            
            data[iteration]['garch'] = {
                'p': p,
                'q': q,
                'mean_model': mean_model,
                'loglikelihood': float(fit_results['loglikelihood']),
                'aic': float(fit_results['aic']),
                'bic': float(fit_results['bic']),
                'params': fit_results['params'],
                'convergence_flag': int(fit_results['convergence_flag']),
                'training_time': training_time
            }
            
            print(f"Completed GARCH({p},{q}) - LL: {fit_results['loglikelihood']:.2f}, AIC: {fit_results['aic']:.2f}, BIC: {fit_results['bic']:.2f}")
        
        except Exception as e:
            data[iteration]['garch'] = {
                'p': p,
                'q': q,
                'mean_model': mean_model,
                'error': str(e)
            }
            print(f"Error fitting GARCH({p},{q}): {str(e)}")
        
        with open(path, "w") as outfile: 
            json.dump(data, outfile, indent=4, default=str)
        
        iteration += 1
    
    if iteration >= n_samples:
        break

print(f"\nResults saved to: {path}")
