import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError("Hessian oracle is not implemented.")

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError("A should be a symmetric matrix.")
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
       func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        mat_vec = self.matvec_Ax(x)
        loss = np.sum(np.logaddexp(0, -self.b * mat_vec)) / self.m
        reg = (self.regcoef / 2) * np.linalg.norm(x) ** 2
        return loss + reg

    def grad(self, x):
        mat_vec = self.matvec_Ax(x)
        weights = expit(-self.b * mat_vec) * (-self.b)
        grad_loss = self.matvec_ATx(weights) / self.m
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        mat_vec = self.matvec_Ax(x)
        p = expit(self.b * mat_vec)
        d = p * (1.0 - p)
        hess_loss = self.matmat_ATsA(d) / self.m
        hess_reg = self.regcoef * np.eye(len(x))
        return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        return super().func_directional(x, d, alpha)

    def grad_directional(self, x, d, alpha):
        return super().grad_directional(x, d, alpha)


def create_log_reg_oracle(A, b, regcoef, oracle_type="usual"):
    """
    Auxiliary function for creating logistic regression oracles.
    """
    if scipy.sparse.issparse(A):
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)

        def matmat_ATsA(s):
            D = scipy.sparse.diags(s)
            return A.T.dot(D).dot(A)

    else:
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        matmat_ATsA = lambda s: (A.T * s).dot(A)

    if oracle_type == "usual":
        oracle = LogRegL2Oracle
    elif oracle_type == "optimized":
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError("Unknown oracle_type=%s" % oracle_type)

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    grad_approx = np.zeros(n)
    f_x = func(x)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        grad_approx[i] = (func(x + eps * e_i) - f_x) / eps

    return grad_approx


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess_approx = np.zeros((n, n))
    f_x = func(x)

    f_eps = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_eps[i] = func(x + eps * e_i)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        for j in range(i, n):
            e_j = np.zeros(n)
            e_j[j] = 1.0

            if i == j:
                val = (func(x + 2.0 * eps * e_i) - 2.0 * f_eps[i] + f_x) / (eps**2)
            else:
                val = (func(x + eps * e_i + eps * e_j) - f_eps[i] - f_eps[j] + f_x) / (
                    eps**2
                )

            hess_approx[i, j] = val
            hess_approx[j, i] = val

    return hess_approx
