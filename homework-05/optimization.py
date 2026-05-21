import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.optimize
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
    """

    def __init__(self, method="Wolfe", **kwargs):
        self._method = method
        if self._method == "Wolfe":
            self.c1 = kwargs.get("c1", 1e-4)
            self.c2 = kwargs.get("c2", 0.9)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Armijo":
            self.c1 = kwargs.get("c1", 1e-4)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Constant":
            self.c = kwargs.get("c", 1.0)
        else:
            raise ValueError("Unknown method {}".format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError("LineSearchTool initializer must be of type dict")
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == "Constant":
            return self.c

        elif self._method == "Armijo":
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = oracle.func(x_k)
            grad_0 = oracle.grad(x_k)
            phi_prime_0 = grad_0.dot(d_k)

            while (
                oracle.func(x_k + alpha * d_k) > phi_0 + self.c1 * alpha * phi_prime_0
            ):
                alpha /= 2.0
            return alpha

        elif self._method == "Wolfe":
            phi_0 = oracle.func(x_k)
            grad_0 = oracle.grad(x_k)
            phi_prime_0 = grad_0.dot(d_k)

            alpha, _, _, _, _, _ = scipy.optimize.linesearch.scalar_search_wolfe2(
                lambda a: oracle.func(x_k + a * d_k),
                lambda a: oracle.grad(x_k + a * d_k).dot(d_k),
                phi_0,
                phi_prime_0,
                self.c1,
                self.c2,
            )

            if alpha is None:
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                while (
                    oracle.func(x_k + alpha * d_k)
                    > phi_0 + self.c1 * alpha * phi_prime_0
                ):
                    alpha /= 2.0

            return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(
    oracle,
    x_0,
    tolerance=1e-5,
    max_iter=10000,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    initial_grad = oracle.grad(x_k)
    initial_grad_norm_sq = np.sum(initial_grad**2)

    for iteration in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)

        if trace:
            history["time"].append((datetime.now() - start_time).total_seconds())
            history["func"].append(oracle.func(x_k))
            history["grad_norm"].append(grad_norm)
            if x_k.size <= 2:
                history["x"].append(np.copy(x_k))

        if grad_norm**2 <= tolerance * initial_grad_norm_sq:
            return x_k, "success", history

        d_k = -grad
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha_k * d_k

    final_grad = oracle.grad(x_k)
    final_grad_norm = np.linalg.norm(final_grad)

    if trace:
        history["time"].append((datetime.now() - start_time).total_seconds())
        history["func"].append(oracle.func(x_k))
        history["grad_norm"].append(final_grad_norm)
        if x_k.size <= 2:
            history["x"].append(np.copy(x_k))

    if final_grad_norm**2 <= tolerance * initial_grad_norm_sq:
        return x_k, "success", history

    return x_k, "iterations_exceeded", history


def newton(
    oracle,
    x_0,
    tolerance=1e-5,
    max_iter=100,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    initial_grad = oracle.grad(x_k)
    initial_grad_norm_sq = np.sum(initial_grad**2)

    for iteration in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)

        if trace:
            history["time"].append((datetime.now() - start_time).total_seconds())
            history["func"].append(oracle.func(x_k))
            history["grad_norm"].append(grad_norm)
            if x_k.size <= 2:
                history["x"].append(np.copy(x_k))

        if grad_norm**2 <= tolerance * initial_grad_norm_sq:
            return x_k, "success", history

        hess = oracle.hess(x_k)
        try:
            c, low = scipy.linalg.cho_factor(hess)
            d_k = scipy.linalg.cho_solve((c, low), -grad)
        except (LinAlgError, ValueError):
            return x_k, "newton_direction_error", history

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        x_k = x_k + alpha_k * d_k

    final_grad = oracle.grad(x_k)
    final_grad_norm = np.linalg.norm(final_grad)

    if trace:
        history["time"].append((datetime.now() - start_time).total_seconds())
        history["func"].append(oracle.func(x_k))
        history["grad_norm"].append(final_grad_norm)
        if x_k.size <= 2:
            history["x"].append(np.copy(x_k))

    if final_grad_norm**2 <= tolerance * initial_grad_norm_sq:
        return x_k, "success", history

    return x_k, "iterations_exceeded", history
