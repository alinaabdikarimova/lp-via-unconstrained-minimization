import numpy as np


def compute_f(x, lambd, s, A, b, c, q = 2.1):

    gamma = c.T @ x - b.T @ lambd
    rho = A @ x - b
    sigma = A.T @ lambd + s - c

    relu_x = np.maximum(0, -x) ** q
    relu_s = np.maximum(0, -s) ** q

    return (
            0.5 * gamma ** 2 +
            0.5 * np.linalg.norm(rho) ** 2 +
            0.5 * np.linalg.norm(sigma) ** 2 +
            (1 / (q*(q-1))) * np.sum(relu_x) +
            (1 / (q*(q-1))) * np.sum(relu_s)
    )


def compute_grad_f(x, lambd, s, A, b, c, q = 2.1):

    gamma = c.T @ x - b.T @ lambd
    rho = b - A @ x
    sigma = (c - A.T @ lambd - s)

    relu_x_grad = np.maximum(0, -x) ** (q-1)
    relu_s_grad = np.maximum(0, -s) ** (q-1)

    grad = (
            gamma * np.concatenate([c, -b, np.zeros_like(s)])
            - np.concatenate([A.T @ rho, A @ sigma, sigma])
            - (1 / (q-1)) * np.concatenate([relu_x_grad, np.zeros_like(b), relu_s_grad])
    )

    return grad


def backtracking_line_search(x, lambd, s, A, b, c, dx, alpha=0.01, beta=0.5, t_init=1.0, q = 2.1):

    m, n = np.shape(A)
    t = t_init
    fx = compute_f(x, lambd, s, A, b, c, q = q)
    grad_fx = compute_grad_f(x, lambd, s, A, b, c, q = q)

    while compute_f(x + t * dx[:n], lambd + t * dx[n:n+m], s + t * dx[n+m:], A, b, c, q = q) > fx + alpha * t * np.dot(grad_fx,dx):
        t *= beta  # Reduce step size

    return t


def construct_hessian(x, s, A, b, c, q = 2.1, mu=1e-5):
    m, n = A.shape
    I_n = np.eye(n)

    H = np.block([
        [np.outer(c, c) + A.T @ A, -np.outer(c, b), np.zeros((n, n))],
        [-np.outer(b, c), np.outer(b, b) + A @ A.T, A],
        [np.zeros((n, n)), A.T, I_n]
    ])

    diag_correction = np.diag(np.hstack([np.maximum(-x, 0) ** (q-2), np.zeros(m), np.maximum(-s, 0) ** (q-2)]))
    H_mod = H + diag_correction + mu * np.eye(m+2*n)

    return H_mod


def wolfe_line_search(x, lambd, s, dx, A, b, c, alpha_max=1.0, c1=1e-4, c2=0.9, max_iters=50, q = 2.01):
    m, n = np.shape(A)
    alpha_low = 0
    alpha_high = alpha_max
    alpha = alpha_high

    fx = compute_f(x, lambd, s, A, b, c, q=q)
    grad_fx = compute_grad_f(x, lambd, s, A, b, c, q=q)

    for _ in range(max_iters):
        x_new = x + alpha * dx[:n]  # Accept the step
        lambd_new = lambd + alpha * dx[n:m+n]
        s_new = s + alpha * dx[m+n:]

        fx_new = compute_f(x_new, lambd_new, s_new, A, b, c, q=q)
        grad_fx_new = compute_grad_f(x_new, lambd_new, s_new, A, b, c, q=q)

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



def ldl_modified(A, beta = 1e-7, delta = 1e-7):
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    C = np.zeros((n,n))

    for j in range(n):
        temp_sum = 0
        for s in range(j):
            temp_sum += L[j, s] ** 2 * D[s]
        C[j,j] = A[j, j] - temp_sum
        if j < n - 1:
            v = np.abs(C[j + 1:n, j])
            theta = np.max(v)
            D[j] = max(abs(C[j,j]), (theta/beta) ** 2 ,delta)
        else:
            D[j] = C[j,j]
        for i in range(j + 1, n):
            temp_sum = 0
            for s in range(j):
                temp_sum += L[i, s] * L[j, s] * D[s]
            C[i, j] = A[i,j] - temp_sum
            L[i, j] = C[i, j]/D[j]

    return L, np.diag(D)
