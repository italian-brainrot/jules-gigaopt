import argparse
import math
import time

import numpy as np

from .mbs import mbs_minimize
from .optimizers import Adam, SGD, Athena, SOAP, LBFGS

# --- Synthetic Objective Definition ---

class SyntheticObjective:
    def __init__(self, n_dims, n_samples, batch_size, correlation_rho=0.9, non_convex_alpha=0.2, non_convex_beta=1):
        self.n_dims = n_dims
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.alpha = non_convex_alpha
        self.beta = non_convex_beta

        # For reproducibility
        np.random.seed(42)

        # Create a covariance matrix with strong off-diagonal elements
        H = np.full((n_dims, n_dims), correlation_rho)
        np.fill_diagonal(H, 1.0)

        # Use Cholesky decomposition to get the transformation matrix
        L = np.linalg.cholesky(H)

        # Create the data matrix A with the desired covariance structure
        Z = np.random.randn(n_samples, n_dims)
        self.A = Z @ L.T

        # Create true parameters and target b
        self.x_true = np.random.randn(n_dims, 1)
        self.b = self.A @ self.x_true + np.random.randn(n_samples, 1) * 0.01

        self.x_init = np.zeros((n_dims, 1))

    def get_mini_batches(self):
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield self.A[batch_indices], self.b[batch_indices]

    def loss_and_grad(self, x, A_batch=None, b_batch=None):
        if A_batch is None:
            A_batch = self.A
            b_batch = self.b

        # Quadratic part
        diff = A_batch @ x - b_batch
        loss = 0.5 * np.mean(diff**2)
        grad_w = A_batch.T @ diff / len(A_batch)

        # Non-convex part
        loss += self.alpha * np.mean(np.cos(self.beta * x))
        grad_w -= self.alpha * self.beta * np.sin(self.beta * x)

        return loss, grad_w

    def full_loss(self, x):
        return self.loss_and_grad(x)[0]

# --- Experiment Runner ---

def run_experiment(optimizer_class, optimizer_params, lr_low=1e-4, lr_high=10):
    # Hyperparameters for the synthetic objective
    N_DIMS = 200
    N_SAMPLES = 1000
    BATCH_SIZE = 32
    EPOCHS = 100

    objective_func = SyntheticObjective(N_DIMS, N_SAMPLES, BATCH_SIZE)
    num_trials = 0

    def objective(lr):
        nonlocal num_trials
        num_trials += 1

        current_params = optimizer_params.copy()
        current_params['lr'] = lr

        optimizer = optimizer_class(**current_params)
        # Optimizers expect a list of parameters
        params = [np.copy(objective_func.x_init)]

        for epoch in range(EPOCHS):
            for A_batch, b_batch in objective_func.get_mini_batches():
                def closure(backward=True):
                    x = params[0]
                    grads = None
                    if backward:
                        loss, grad_w = objective_func.loss_and_grad(x, A_batch, b_batch)
                        grads = [grad_w]
                    else:
                        loss = objective_func.loss_and_grad(x, A_batch, b_batch)[0]

                    return loss, grads

                optimizer.step(params, closure)

        final_loss = objective_func.full_loss(params[0])
        return final_loss

    grid = np.linspace(math.log10(lr_low), math.log10(lr_high), 6)
    ret = mbs_minimize(objective, grid=grid, num_binary=6, step=1, log_scale=True)

    trials = sorted([(lr, loss[0]) for lr,loss in ret.items()], key=lambda x: x[1])
    best_lr, best_value = trials[0]

    print(f"Best LR for {optimizer_class.__name__}: {best_lr:.5f}")
    print(f"Best value: {best_value:.5f}")
    print("---")

    return best_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true', help='Rerun all experiments')
    args = parser.parse_args()

    if args.rerun:
        start_time = time.time()
        losses = {
            "SGD": run_experiment(SGD, {}),
            "Momentum": run_experiment(SGD, {"momentum": 0.9}),
            "Adam": run_experiment(Adam, {}),
            "Athena": run_experiment(Athena, {}),
            "SOAP": run_experiment(SOAP, {}),
            "LBFGS": run_experiment(LBFGS, {}),
        }
        print(f"Total experiment time: {time.time() - start_time:.2f}s")
    else:
        # Cached results from the last run
        losses = {
            'SOAP': 0.84929,
            'Athena': 0.85200,
            'Adam': 1.24993,
            'Momentum': 3.62479,
            'SGD': 3.63521,
            'LBFGS': 4.68933,
        }

    print("\nOptimizer Performance Comparison on Synthetic Objective")
    # Sort by performance
    sorted_losses = sorted(losses.items(), key=lambda item: item[1])
    for optimizer, final_loss in sorted_losses:
        print(f'{optimizer:<30} {final_loss:.5f}')

if __name__ == "__main__":
    main()