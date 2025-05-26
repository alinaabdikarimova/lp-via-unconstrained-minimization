import numpy as np


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
            nu * (1 / (q*(q-1))) * np.sum(relu_s_pos)
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
    )

    return grad


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

    H = np.block([
        [np.outer(c, c) + A.T @ A, -np.outer(c, b), np.zeros((n, n))],
        [-np.outer(b, c), np.outer(b, b) + A @ A.T, A],
        [np.zeros((n, n)), A.T, I_n]
    ])

    diag_correction = np.diag(np.hstack([np.maximum(-x, 0) ** (q-2), np.zeros(m), np.maximum(-s, 0) ** (q-2)]))
    diag_correction_pos = np.diag(np.hstack([np.maximum(x, 0) ** (q - 2), np.zeros(m), np.maximum(s, 0) ** (q - 2)]))

    H_mod = H + diag_correction + nu * diag_correction_pos + mu * np.eye(m+2*n)

    return H_mod


def wolfe_line_search(x, lambd, s, dx, A, b, c, alpha_max=1.0, c1=1e-4, c2=0.9, max_iters=50, q = 2.1, nu = 1.0):
    m, n = np.shape(A)
    alpha_low = 0
    alpha_high = alpha_max
    alpha = alpha_high

    fx = compute_f(x, lambd, s, A, b, c, q=q, nu = nu)
    grad_fx = compute_grad_f(x, lambd, s, A, b, c, q=q, nu = nu)

    for _ in range(max_iters):
        x_new = x + alpha * dx[:n]  # Accept the step
        lambd_new = lambd + alpha * dx[n:m+n]
        s_new = s + alpha * dx[m+n:]

        fx_new = compute_f(x_new, lambd_new, s_new, A, b, c, q=q, nu = nu)
        grad_fx_new = compute_grad_f(x_new, lambd_new, s_new, A, b, c, q=q, nu = nu)

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

