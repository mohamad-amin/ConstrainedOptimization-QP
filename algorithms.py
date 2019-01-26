import numpy as np
import numpy.linalg as la


def barrier_move_not_acceptable_for_qp(P, q, x, residual, f_val, v, f_prime, d_residual, s, t, alpha):
    moved = x + s * v
    moved_val = t * (.5 * np.matmul(np.matmul(moved.T, P), moved) + np.matmul(q.T, moved)) \
        - np.sum(np.log(residual + s * d_residual))
    return moved_val >= f_val + alpha * s * f_prime


def log_barrier_for_qp(P, q, A, b, alpha, beta, mu, iterations, tol=1e-3, eps=1e-6):

    t = 1
    gap = np.inf
    m = A.shape[0]
    n = A.shape[1]
    x = np.zeros((n, 1))
    gap_results = []

    for iteration in range(iterations):

        residual = (b - np.matmul(A, x)).astype('f')
        f_val = t * (.5 * np.matmul(x.T, np.matmul(P, x)) + np.matmul(q.T, x)) - np.sum(np.log(residual))
        f_grad = t * (np.matmul(P, x) + q) + np.matmul(A.T, np.reciprocal(residual))
        f_hess = t * P + np.matmul(A.T, np.matmul(np.diag(np.reciprocal(np.square(residual))[:, 0]), A))

        v = -1 * la.lstsq(f_hess, f_grad, rcond=None)[0]
        f_prime = np.matmul(f_grad.T, v)
        d_residual = -np.matmul(A, v)
        s = 1
        while np.min(residual + s * d_residual) <= 0:
            s *= beta
        while barrier_move_not_acceptable_for_qp(P, q, x, residual, f_val, v, f_prime, d_residual, s, t, alpha):
            s *= beta
            if s == 0:
                break

        x += s * v
        if -f_prime < eps:
            gap = m / t
            if gap < tol:
                print('Iteration: %d - final gap: %f' % (iteration, gap))
                gap_results.append(gap)
                break
            t = mu * t
        gap_results.append(gap)

    return gap_results


def calculate_new_r(P, q, A, t_inv, step, z, dz, x, dx, s, ds):
    newz = z + step * dz
    newx = x + step * dx
    news = s + step * ds
    return np.concatenate((np.matmul(P, newx) + q + np.matmul(A.T, newz), newz * news - t_inv), axis=0)


def interior_point_for_qp(P, q, A, b, alpha, beta, mu, iterations, tol=1e-3, eps=1e-6):

    m = A.shape[0]
    n = A.shape[1]
    x = np.zeros((n, 1))
    s = (b - np.matmul(A, x)).astype('f')
    z = np.reciprocal(s)
    surrogates = []
    dual_residual_norms = []

    for iteration in range(iterations):

        gap = np.asscalar(np.matmul(s.T, z))
        res = np.matmul(P, x) + q + np.matmul(A.T, z)
        surrogates.append(gap)
        dual_residual_norms.append(la.norm(res))
        if (gap < tol) and (la.norm(res) < eps):
            break

        t_inv = gap / (m * mu)
        sol_a = -1 * np.concatenate(
            (np.concatenate((P, A.T), axis=1), np.concatenate((A, np.diag(-np.divide(s, z)[:, 0])), axis=1)), axis=0)
        sol_b = np.concatenate((res, -s + t_inv * np.reciprocal(z)), axis=0)
        sol = la.lstsq(sol_a, sol_b, rcond=None)[0]

        dx = sol[0:n, :]
        dz = sol[n:n+m, :]
        ds = np.matmul(-A, dx)
        r = np.concatenate((res, z*s - t_inv), axis=0)

        step = min(1.0, 0.99 / np.max(np.divide(-dz, z)))
        while np.min(s + step * ds) <= 0:
            step *= beta
            if step == 0:
                break
        while la.norm(calculate_new_r(P, q, A, t_inv, step, z, dz, x, dx, s, ds)) > (1-alpha*step) * la.norm(r):
            step *= beta
            if step == 0:
                break

        x += step * dx
        z += step * dz
        s = b - np.matmul(A, x)

    return surrogates, dual_residual_norms
