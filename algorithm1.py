import numpy as np
from scipy.optimize import linprog
from algorithm1_functions import compute_f, compute_grad_f, backtracking_line_search, construct_hessian, wolfe_line_search
from scipy.linalg import cholesky, solve_triangular
from numpy.linalg import inv
import math

def newtons_method(x0, lambd0, s0, A, b, c, q = 2.1, mu = 1e-9, max_iter=1000, tol = 1e-6, x_star = 0):

    x = x0
    lambd = lambd0
    s = s0

    n_of_systems = 0

    gamma = c.T @ x - b.T @ lambd
    rho = b - A @ x
    sigma = c - A.T @ lambd - s
    min_entry = min(min(x), min(s))

    for i in range(max_iter):

        g = compute_grad_f(x, lambd, s, A, b, c, q = q)
        f = compute_f(x, lambd, s, A, b, c, q=q)
        # H = construct_hessian(x, s, A, b, c, q = q, mu = math.sqrt(np.linalg.norm(g)/2)) #Algorithm 1a
        H = construct_hessian(x, s, A, b, c, q=q, mu=mu) #Algorithm 1b

        p = np.linalg.solve(H, -g)

        n_of_systems += 1

        t = backtracking_line_search(x, lambd, s, A, b, c, p, alpha=0.01, beta=0.7, t_init=1.0, q = q)
        # t = wolfe_line_search(x, lambd, s, p, A, b, c, alpha_max=1.0, c1=1e-4, c2=0.9, max_iters=50, q = q)
        # t = 1

        x += t*p[:n]
        lambd += t*p[n:m + n]
        s += t*p[m + n:]
        
        gamma = c.T @ x - b.T @ lambd
        rho = b - A @ x
        sigma = c - A.T @ lambd - s
        min_entry = min(min(x), min(s))
        
        if i % 1 == 0:
            print(f"--------------Iteration{i}-----------------")
            print(f"Norm of gradient is: {np.linalg.norm(g)}")
            print(f"Objective function value is: {np.linalg.norm(f)}")
            print(f"Norm of gamma: {np.linalg.norm(gamma)}")
            print(f"Norm of rho: {np.linalg.norm(rho)}")
            print(f"Norm of sigma: {np.linalg.norm(sigma)}")
            print(f"min entry: {min_entry}")
            print("Residual: ", np.linalg.norm(x_star - x) / np.linalg.norm(x_star))
            print(f"Linear systems solved: {n_of_systems}")
            
        if gamma < tol and np.linalg.norm(rho) < tol and np.linalg.norm(sigma) < tol and min_entry > -tol:
            print(f"We converged at iteration {i}")
            return x

    return x


np.random.seed(1)
m, n = 100, 150

# Linear program with an optimal solution
x_star = np.round(10 * np.random.rand(n))
k = n - m
indices_to_zero = np.random.choice(n, size=k, replace=False)
x_star[indices_to_zero] = 0
y_star = np.round(10 * np.random.randn(m))
z_star = np.round(10 * np.random.rand(n))
z_star[x_star != 0] = 0
A = np.round(10*np.random.randn(m, n))
b = A @ x_star
c = A.T @ y_star + z_star

# Unbounded linear program
# x_star = np.round(10 * np.random.rand(n))
# A = np.round(10 * np.random.randn(m, n))
# b = A @ x_star
# c = np.round(10 * np.random.rand(n) - 5)

x = np.zeros(n)
lambd = np.zeros(m)
s = np.zeros(n)

solution = newtons_method(x, lambd, s, A, b, c, q = 2.1, mu = 1e-9, max_iter=1000, tol = 1e-8, x_star = x_star)


