import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg import block_diag

def compute_f(x, lambd, s, A, b, c, q = 2.1, nu = 1.0):

    gamma = c.T @ x - b.T @ lambd
    rho = A @ x - b
    sigma = A.T @ lambd + s - c

    relu_x = np.maximum(0, -x) ** q
    relu_s = np.maximum(0, -s) ** q

    relu_x_pos = np.maximum(0, x) ** q
    relu_s_pos = np.maximum(0, s) ** q

    return (
            0.5 * gamma ** 2 +
            0.5 * np.linalg.norm(rho) ** 2 +
            0.5 * np.linalg.norm(sigma) ** 2 +
            (1 / (q*(q-1))) * np.sum(relu_x) +
            (1 / (q*(q-1))) * np.sum(relu_s) +
            nu * (1 / (q*(q-1))) * np.sum(relu_x_pos) +
            nu * (1 / (q*(q-1))) * np.sum(relu_s_pos) +
            0.5 * nu * np.linalg.norm(lambd) ** 2
    )


def compute_grad_f(x, lambd, s, A, b, c, q = 2.1, nu = 1.0):

    gamma = c.T @ x - b.T @ lambd
    rho = b - A @ x
    sigma = (c - A.T @ lambd - s)

    relu_x_grad = np.maximum(0, -x) ** (q - 1)
    relu_s_grad = np.maximum(0, -s) ** (q - 1)
    relu_x_pos_grad = nu * np.maximum(0, x) ** (q - 1)
    relu_s_pos_grad = nu * np.maximum(0, s) ** (q - 1)


    grad = (
            gamma * np.concatenate([c, -b, np.zeros_like(s)])
            - np.concatenate([A.T @ rho, A @ sigma, sigma])
            - (1 / (q - 1)) * np.concatenate([relu_x_grad, np.zeros_like(b), relu_s_grad])
            + (1 / (q - 1)) * np.concatenate([relu_x_pos_grad, np.zeros_like(b), relu_s_pos_grad])
            + nu * np.concatenate([np.zeros_like(c), lambd, np.zeros_like(c)])
    )

    return grad


def solve_system(x, lambd, s, A, b, c, q = 2.1, nu = 1.0, mu = 1e-5):
    n = len(x)
    m = len(lambd)

    p = np.zeros(2*n+m)

    grad = compute_grad_f(x, lambd, s, A, b, c, q = q, nu = nu)

    g_x = grad[:n]
    g_lambda = grad[n:m+n]
    g_s = grad[m+n:]

    D1 = np.maximum(-x, 0) ** (q-2) + nu * np.maximum(x, 0) ** (q-2) + mu * np.ones(n)
    D2 = (nu + mu)* np.ones(m)
    D3 = np.maximum(-s, 0) ** (q-2) + nu * np.maximum(s, 0) ** (q-2) + mu * np.ones(n)

    Gram_matrix = A.T @ A
    Gram_matrix[np.diag_indices_from(Gram_matrix)] += D1

    Gram_matrix_mod = A @ np.diag(D3/(1+D3)) @ A.T
    Gram_matrix_mod[np.diag_indices_from(Gram_matrix_mod)] += D2

    L_lambda = cholesky(Gram_matrix_mod, lower=True)
    L_x = cholesky(Gram_matrix, lower=True)

    L_x_lambda = block_diag(L_x, L_lambda)

    c = c.reshape(-1, 1)
    b = b.reshape(-1, 1)
    u = np.vstack([c, -b])
    u = u.flatten()

    v = np.concatenate([
        -g_x,
        A @ (g_s / (1 + D3)) - g_lambda
    ])

    L_x_lambda_updated = cholesky_rank1_update(L_x_lambda, u)
    y_x_lambda = solve_triangular(L_x_lambda_updated, v, lower=True)
    p[:m+n] = solve_triangular(L_x_lambda_updated.T, y_x_lambda, lower=False)

    p[m+n:] = (-g_s-A.T @ p[n:m+n])/(1+D3)

    return p


def cholesky_rank1_update(L, u):
    """
    Perform a rank-1 Cholesky update.

    Parameters:
    L : ndarray
        Lower-triangular Cholesky factor of A, where A = L @ L.T
    u : ndarray
        Vector u, so that A_new = A + u @ u.T

    Returns:
    L_new : ndarray
        Updated lower-triangular Cholesky factor of A + u @ u.T
    """
    L = L.copy()
    n = len(u)
    u = u.copy()

    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + u[k] ** 2)
        c = r / L[k, k]
        s = u[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k + 1:, k] = (L[k + 1:, k] + s * u[k + 1:]) / c
            u[k + 1:] = c * u[k + 1:] - s * L[k + 1:, k]

    return L


def backtracking_line_search(x, lambd, s, A, b, c, dx, alpha=0.01, beta=0.5, t_init=1.0, q = 2.1, nu = 1.0):

    m, n = np.shape(A)
    t = t_init
    fx = compute_f(x, lambd, s, A, b, c, q = q, nu = nu)
    grad_fx = compute_grad_f(x, lambd, s, A, b, c, q = q, nu = nu)

    while compute_f(x + t * dx[:n], lambd + t * dx[n:n+m], s + t * dx[n+m:], A, b, c, q = q, nu = nu) > fx + alpha * t * np.dot(grad_fx,dx):
        t *= beta  # Reduce step size
    return t


def construct_hessian(x, s, A, b, c, q = 2.1, mu=1e-5, nu = 1.0):
    m, n = A.shape
    I_n = np.eye(n)
    I_m = np.eye(m)

    H = np.block([
        [np.outer(c, c) + A.T @ A, -np.outer(c, b), np.zeros((n, n))],
        [-np.outer(b, c), np.outer(b, b) + A @ A.T, A],
        [np.zeros((n, n)), A.T, I_n]
    ])

    diag_correction = np.diag(np.hstack([np.maximum(-x, 0) ** (q-2), np.zeros(m), np.maximum(-s, 0) ** (q-2)]))
    diag_correction_pos = np.diag(np.hstack([np.maximum(x, 0) ** (q - 2), np.ones(m), np.maximum(s, 0) ** (q - 2)]))

    H_mod = H + diag_correction + nu * diag_correction_pos + mu * np.eye(m+2*n)

    return H_mod


def wolfe_line_search(x, lambd, s, dx, A, b, c, alpha_max=1.0, c1=1e-4, c2=0.9, max_iters=50, q = 2.1, nu = 1.0):
    m, n = np.shape(A)
    alpha_low = 0
    alpha_high = alpha_max
    alpha = alpha_high

    fx = compute_f(x, lambd, s, A, b, c, q = q, nu = nu)
    grad_fx = compute_grad_f(x, lambd, s, A, b, c, q = q, nu = nu)

    for _ in range(max_iters):
        x_new = x + alpha * dx[:n]  # Accept the step
        lambd_new = lambd + alpha * dx[n:m+n]
        s_new = s + alpha * dx[m+n:]

        fx_new = compute_f(x_new, lambd_new, s_new, A, b, c, q = q, nu = nu)
        grad_fx_new = compute_grad_f(x_new, lambd_new, s_new, A, b, c, q = q, nu = nu)

        # Check Armijo condition (sufficient decrease)
        if fx_new > fx + c1 * alpha * np.dot(grad_fx, dx):
            alpha_high = alpha  # Reduce step size
        # Check Curvature condition (Wolfe condition)
        elif np.dot(grad_fx_new, dx) < c2 * np.dot(grad_fx, dx):
            alpha_low = alpha  # Increase step size
        else:
            return alpha  # Found valid step size

        # Update alpha using bisection
        alpha = 0.5 * (alpha_low + alpha_high)

    return alpha  # Return best alpha found

