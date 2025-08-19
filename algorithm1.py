import numpy as np
from scipy.optimize import linprog
from functions import compute_f, compute_grad_f, backtracking_line_search, construct_hessian, wolfe_line_search, solve_system
from scipy.linalg import cholesky, solve_triangular
from numpy.linalg import inv
import math
import os
import zipfile

def newtons_method(x0, lambd0, s0, A, b, c, q = 2.1, theta = 0.1, mu = 1e-9, max_iter=1000, tol = 1e-6, nu0 = 0, x_star = 0):

    x = x0
    lambd = lambd0
    s = s0
    
    nu = nu0

    gamma = c.T @ x - b.T @ lambd
    rho = b - A @ x
    sigma = c - A.T @ lambd - s
    min_entry = min(min(x), min(s))

    for i in range(max_iter):

        g = compute_grad_f(x, lambd, s, A, b, c, q = q,nu = nu)
        f = compute_f(x, lambd, s, A, b, c, q=q, nu = nu)

        'Algorithm 1a'
        # mu = math.sqrt(np.linalg.norm(g)/2)
        
        'Algorithm 1b'
        mu = 1e-9

        p = solve_system(x, lambd, s, A, b, c, q = q, nu = nu, mu = mu)

        'Algorithm 1a'
        # t = 1
        
        'Algorithm 1b'
        t = backtracking_line_search(x, lambd, s, A, b, c, p, alpha=0.01, beta=0.5, t_init=1.0, q = q, nu = nu)
        # t = wolfe_line_search(x, lambd, s, p, A, b, c, alpha_max=1.0, c1=1e-4, c2=0.9, max_iters=50, q = q, nu = nu)

        x += t*p[:n] 
        lambd += t*p[n:m + n]
        s += t*p[m + n:]

        gamma = c.T @ x - b.T @ lambd
        rho = b - A @ x
        sigma = c - A.T @ lambd - s
        min_entry = min(min(x), min(s))

        nu = nu * theta

        if i % 1 == 0:
            print(f"--------------Iteration{i}-----------------")
            print(f"Norm of gradient: {np.linalg.norm(g)}")
            print(f"Objective function value: {np.linalg.norm(f)}")
            print(f"Absolute value of gamma: {np.linalg.norm(gamma)}")
            print(f"Norm of rho: {np.linalg.norm(rho)}")
            print(f"Norm of sigma: {np.linalg.norm(sigma)}")
            print(f"min{{x_j, s_j}}: {min_entry}")
            print("Relative error of x_k with respect to x_star: ", np.linalg.norm(x_star - x) / np.linalg.norm(x_star))

        # stopping criterion based on optimality conditions    
        # if gamma < tol and np.linalg.norm(rho) < tol and np.linalg.norm(sigma) < tol and min_entry > -tol:

        # stopping criterion based on relative error to known solution x_star
        if np.linalg.norm(x_star - x)/np.linalg.norm(x_star) < tol:
            print(f"We converged at iteration {i}")
            return x
    return x


# ----------------------------------------
# Choose the test problem to solve:
# ----------------------------------------
# 1 - Linear program with an optimal solution (m = 100, n = 150)
# 2 - Linear program with an optimal solution (m = 200, n = 300)
# 3 - Linear program with an optimal solution (m = 500, n = 750)
# 4 - Unbounded linear program (m = 50, n = 150)
# ----------------------------------------

problem_id = 3  # Change this to 2, 3, or 4 to solve a different problem

zip_path = "problem_set.zip"
extract_dir = "problem_set"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"Extracted files to '{extract_dir}'")

def load_csv(name):
    return np.loadtxt(os.path.join(extract_dir, f"problem{problem_id}_{name}.csv"), delimiter=",")

A = load_csv("A")
b = load_csv("b")
c = load_csv("c")
x_star = load_csv("x_star")
m, n = A.shape

# Initial point
x = np.zeros(n)
lambd = np.zeros(m)
s = np.zeros(n)

solution = newtons_method(x, lambd, s, A, b, c, q = 2.1, theta = 0.8, mu = 1e-9, max_iter=1000, tol = 1e-9, nu0 = 0, x_star = x_star)

# -------------------------------
# Random Problem Setup
# -------------------------------

# Generate a random linear program with a known optimal solution

# m, n = 100, 150
# x_star = np.round(10 * np.random.rand(n))
# k = n - m
# indices_to_zero = np.random.choice(n, size=k, replace=False)
# x_star[indices_to_zero] = 0
# y_star = np.round(10 * np.random.randn(m))
# z_star = np.round(10 * np.random.rand(n))
# z_star[x_star != 0] = 0
# A = np.round(10*np.random.randn(m, n))
# b = A @ x_star
# c = A.T @ y_star + z_star

# Alternatively, generate a potentially unbounded linear program

# m, n = 50, 150
# x_star = np.round(10 * np.random.rand(n))
# A = np.round(10 * np.random.randn(m, n))
# b = A @ x_star
# c = np.round(10 * np.random.rand(n) - 5)
