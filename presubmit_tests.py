import unittest
import numpy.testing as npt

# Заглушки, чтобы не использовать старую библиотеку nose
assert_almost_equal = npt.assert_almost_equal
def ok_(expr, msg=None): assert expr, msg
def eq_(a, b, msg=None): assert a == b, msg
def attr(*args, **kwargs): return lambda f: f

from io import StringIO
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import sys
import warnings

import optimization
import oracles


def check_log_reg(oracle_type, sparse=False):
    A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if sparse:
        A = scipy.sparse.csr_matrix(A)
    b = np.array([1, 1, -1, 1])
    reg_coef = 0.5
    logreg = oracles.create_log_reg_oracle(A, b, reg_coef, oracle_type=oracle_type)

    x = np.zeros(2)
    assert_almost_equal(logreg.func(x), 0.693147180)
    ok_(np.allclose(logreg.grad(x), [0, -0.25]))
    ok_(np.allclose(logreg.hess(x), [[0.625, 0.0625], [0.0625, 0.625]]))
    ok_(isinstance(logreg.grad(x), np.ndarray))
    ok_(isinstance(logreg.hess(x), np.ndarray))

    x = np.zeros(2)
    d = np.ones(2)
    assert_almost_equal(logreg.func_directional(x, d, alpha=0.5), 0.7386407091095)
    assert_almost_equal(logreg.grad_directional(x, d, alpha=0.5), 0.4267589549159)
    assert_almost_equal(logreg.func_directional(x, d, alpha=1.0), 1.1116496416598)
    assert_almost_equal(logreg.grad_directional(x, d, alpha=1.0), 1.0559278283039)


def get_counters(A):
    counters = {"Ax": 0, "ATx": 0, "ATsA": 0}
    def matvec_Ax(x):
        counters["Ax"] += 1
        return A.dot(x)
    def matvec_ATx(x):
        counters["ATx"] += 1
        return A.T.dot(x)
    def matmat_ATsA(s):
        counters["ATsA"] += 1
        return A.T.dot(A * s.reshape(-1, 1))
    return (matvec_Ax, matvec_ATx, matmat_ATsA, counters)


def check_counters(counters, groundtruth):
    for key, value in groundtruth.items():
        ok_(key in counters)
        ok_(counters[key] <= value)


def check_equal_histories(history1, history2, atol=1e-3):
    if history1 is None or history2 is None:
        eq_(history1, history2)
        return
    ok_("func" in history1 and "func" in history2)
    ok_(np.allclose(history1["func"], history2["func"], atol=atol))
    ok_("grad_norm" in history1 and "grad_norm" in history2)
    ok_(np.allclose(history1["grad_norm"], history2["grad_norm"], atol=atol))
    ok_("time" in history1 and "time" in history2)
    eq_(len(history1["time"]), len(history2["time"]))
    eq_("x" in history1, "x" in history2)
    if "x" in history1:
        ok_(np.allclose(history1["x"], history2["x"], atol=atol))


def check_prototype(method):
    class ZeroOracle2D(oracles.BaseSmoothOracle):
        def func(self, x): return 0.0
        def grad(self, x): return np.zeros(2)
        def hess(self, x): return np.zeros([2, 2])

    oracle = ZeroOracle2D()
    x0 = np.ones(2)
    HISTORY = {
        "func": [0.0],
        "grad_norm": [0.0],
        "time": [0],
        "x": [np.ones(2)],
    }

    def check_result(result, x0=np.ones(2), msg="success", history=None):
        eq_(len(result), 3)
        ok_(np.allclose(result[0], x0))
        eq_(result[1], msg)
        check_equal_histories(result[2], history)

    check_result(method(oracle, x0))
    check_result(method(oracle, x0, 1e-3, 10))
    check_result(method(oracle, x0, 1e-3, 10, {"method": "Constant", "c": 1.0}))
    check_result(
        method(oracle, x0, 1e-3, 10, {"method": "Constant", "c": 1.0}, trace=True),
        history=HISTORY,
    )
    check_result(
        method(
            oracle, x0, 1e-3, max_iter=10,
            line_search_options={"method": "Constant", "c": 1.0}, trace=True, display=True
        ),
        history=HISTORY,
    )
    check_result(method(oracle, x0, display=True, trace=False))
    check_result(method(oracle, x0, tolerance=1e-8, trace=True), history=HISTORY)

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    check_result(method(oracle, x0))
    eq_(mystdout.getvalue(), "")
    sys.stdout = old_stdout

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    check_result(method(oracle, x0, display=False))
    eq_(mystdout.getvalue(), "")
    sys.stdout = old_stdout

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    check_result(method(oracle, x0, display=True))
    ok_(len(mystdout.getvalue()) > 1)
    sys.stdout = old_stdout


def check_one_ideal_step(method):
    oracle = get_quadratic()
    x0 = np.ones(3) * 10.0
    [x_star, msg, history] = method(oracle, x0, max_iter=1, tolerance=1e-5, trace=True)
    ok_(np.allclose(x_star, [1.0, 2.0, 3.0]))
    eq_(msg, "success")
    check_equal_histories(
        history,
        {
            "func": [90.0, -7.0],
            "grad_norm": [13.928388277184119, 0.0],
            "time": [0, 1],
        },
    )


def get_quadratic():
    A = np.eye(3)
    b = np.array([1, 2, 3])
    return oracles.QuadraticOracle(A, b)


def get_1d(alpha):
    class Func(oracles.BaseSmoothOracle):
        def __init__(self, alpha):
            self.alpha = alpha

        def func(self, x):
            return np.exp(self.alpha * x) + self.alpha * x**2

        def grad(self, x):
            return np.array(self.alpha * np.exp(self.alpha * x) + 2 * self.alpha * x)

        def hess(self, x):
            return np.array([self.alpha**2 * np.exp(self.alpha * x) + 2 * self.alpha])
    return Func(alpha)


# Основной класс тестирования
class TestPresubmitTests(unittest.TestCase):
    def test_python3(self):
        ok_(sys.version_info > (3, 0))

    def test_QuadraticOracle(self):
        A = np.eye(3)
        b = np.array([1, 2, 3])
        quadratic = oracles.QuadraticOracle(A, b)

        x = np.zeros(3)
        assert_almost_equal(quadratic.func(x), 0.0)
        ok_(np.allclose(quadratic.grad(x), -b))
        ok_(np.allclose(quadratic.hess(x), A))
        ok_(isinstance(quadratic.grad(x), np.ndarray))
        ok_(isinstance(quadratic.hess(x), np.ndarray))

        x = np.ones(3)
        assert_almost_equal(quadratic.func(x), -4.5)
        ok_(np.allclose(quadratic.grad(x), x - b))
        ok_(np.allclose(quadratic.hess(x), A))
        ok_(isinstance(quadratic.grad(x), np.ndarray))
        ok_(isinstance(quadratic.hess(x), np.ndarray))

        x = np.ones(3)
        d = -np.ones(3)
        assert_almost_equal(quadratic.func_directional(x, d, alpha=0.5), -2.625)
        assert_almost_equal(quadratic.grad_directional(x, d, alpha=0.5), 4.5)
        assert_almost_equal(quadratic.func_directional(x, d, alpha=1.0), 0.0)
        assert_almost_equal(quadratic.grad_directional(x, d, alpha=1.0), 6.0)

    def test_log_reg_usual(self):
        check_log_reg("usual")
        check_log_reg("usual", sparse=True)

    @attr("bonus")
    def test_log_reg_optimized(self):
        return  # ПРОПУСКАЕМ БОНУСНУЮ ЧАСТЬ
        check_log_reg("optimized")
        check_log_reg("optimized", sparse=True)

    def test_log_reg_oracle_calls(self):
        A = np.ones((2, 2))
        b = np.ones(2)
        x = np.ones(2)
        d = np.ones(2)
        reg_coef = 0.5

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func(x)
        check_counters(counters, {"Ax": 1, "ATx": 0, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad(x)
        check_counters(counters, {"Ax": 1, "ATx": 1, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).hess(x)
        check_counters(counters, {"Ax": 1, "ATx": 0, "ATsA": 1})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func_directional(x, d, 1)
        check_counters(counters, {"Ax": 1, "ATx": 0, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad_directional(x, d, 1)
        check_counters(counters, {"Ax": 1, "ATx": 1, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
        oracle.func(x)
        oracle.grad(x)
        oracle.hess(x)
        check_counters(counters, {"Ax": 3, "ATx": 1, "ATsA": 1})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
        oracle.func(x)
        oracle.grad(x)
        check_counters(counters, {"Ax": 2, "ATx": 1, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
        oracle.grad(x)
        oracle.hess(x)
        check_counters(counters, {"Ax": 2, "ATx": 1, "ATsA": 1})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
        oracle.func(x)
        oracle.grad(x)
        oracle.func_directional(x, d, 1)
        oracle.grad_directional(x, d, 2)
        oracle.func_directional(x, d, 2)
        oracle.func_directional(x, d, 3)
        check_counters(counters, {"Ax": 6, "ATx": 2, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
        oracle.func(x)
        oracle.grad(x)
        oracle.func_directional(x, d, 1)
        oracle.grad_directional(x, d, 2)
        oracle.func_directional(x, d, 2)
        oracle.func_directional(x, d, 3)
        oracle.func(x + 3 * d)
        oracle.grad(x + 3 * d)
        check_counters(counters, {"Ax": 8, "ATx": 3, "ATsA": 0})

    @attr("bonus")
    def test_log_reg_optimized_oracle_calls(self):
        return  # ПРОПУСКАЕМ БОНУСНУЮ ЧАСТЬ
        A = np.ones((2, 2))
        b = np.ones(2)
        x = np.ones(2)
        d = np.ones(2)
        reg_coef = 0.5

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func(x)
        check_counters(counters, {"Ax": 1, "ATx": 0, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad(x)
        check_counters(counters, {"Ax": 1, "ATx": 1, "ATsA": 0})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).hess(x)
        check_counters(counters, {"Ax": 1, "ATx": 0, "ATsA": 1})

        matvec_Ax, matvec_ATx, matmat_ATsA, counters = get_counters(A)
        oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func_directional(x, d, 1)