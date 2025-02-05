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

def calculate_mse(theta, theta_true):
    '''The `calculate_mse` function calculates the mean squared error between two parameter
    vectors of the same length
    :param theta: A list or np.ndarray of parameters that are trained.
    :param theta_true: A list or np.ndarray of parameters that were used to generate the
    training data.'''
    mse = 0
    for param, true_param in zip(theta, theta_true):
        mse += (param-true_param)**2
    mse /= len(theta)
    return mse