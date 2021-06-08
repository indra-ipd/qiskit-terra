from typing import Optional

import numpy as np
from numpy import Inf
from scipy.optimize import OptimizeResult
from .scipy_optimizer import SciPyOptimizer
from scipy.optimize.optimize import (
    _prepare_scalar_function,
    _check_unknown_options,
    vecnorm,
    _status_message,
    _line_search_wolfe12,
    _LineSearchError,
)


class aSNAQ(SciPyOptimizer):
    """
    aSNAQ

    See https://www.jstage.jst.go.jp/article/nolta/8/4/8_289/_pdf
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "mu", "lineSearch", "analytical_grad"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: Optional[int] = None,
        maxfev: int = 1024,
        disp: bool = False,
        mu: float = 0.9,
        lineSearch: str = "armijo",
        analytical_grad: bool = True,
        options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Built out using scipy framework, for details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter: Maximum number of iterations to perform.
            maxfev: Maximum number of function evaluations to perform.
            disp: disp
            reset_interval: The minimum estimates directly once
                            in ``reset_interval`` times.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.



        References:
            .. [1] Ninomiya, Hiroshi. "A novel quasi-Newton-based optimization for neural network training
            incorporating Nesterov's accelerated gradient." Nonlinear Theory and Its Applications,
            IEICE 8.4 (2017): 289-301.

        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method=asnaq, options=options, **kwargs)

def vecnorm(x, ord=2):
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)

# pylint: disable=invalid-name

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


def asnaq(
    fun,
    x0,
    args=(),
    jac=None,
    callback=None,
    mu=0.1,
    global_conv=True,
    gtol=1e-5,
    norm=Inf,
    eps=1e-8,
    maxiter=None,
    lineSearch="armijo",
    dirNorm = False,
    disp=False,
    return_all=False,
    finite_diff_rel_step=None,
    gamma=1e-5,
    analytical_grad=False,
    **unknown_options,
):
    """
    Minimization of scalar function of one or more variables using the
    aSNAQ algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    global_conv : bool , default True
        include global convergence term
    lineSearch : str , default: 'armijo', options : 'armijo','wolfe','explicit'
        LineSearch strategies for determining step size
    mu : float/str , options : float: 0 >= mu <1, str: 'adaptive'
        momentum parameter
    gamma : parameter used in adaptive mu, default : gamma = 1e-5

    """


    """
       Bk = minibatch
       |Bk| = b batch size
       L = 5 memory size chosen from (2,5,10,20)
       alpha = ?
       k = iteration count
       mL = 10
       mF = 100
       eps =1e-4
       gamma = 1.01
       """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = np.asarray(x0).flatten()
    wk = x0

    #t = t_vec[0]
    k = 0
    L = 5
    t = 0
    # eps = 1e-4
    # gamma = 1.01
    N = len(wk)

    import collections
    mL = 10
    mF = 100
    mu_fac = 1.01
    mu_init=0.5
    mu_clip=0.99
    warnflag = 0

    if k == 0:
        wo_bar = np.zeros_like(wk)
        vo_bar = np.zeros_like(wk)
        ws = np.zeros_like(wk)
        vs = np.zeros_like(wk)
        vk = np.zeros_like(wk)
        alpha_k = [1]
        sk_vec = collections.deque(maxlen=mL)
        yk_vec = collections.deque(maxlen=mL)
        F = collections.deque(maxlen=mF)
        mu = mu_init

    """
    else:
        wo_bar = wo_bar_vec[0]  # np.zeros_like(wk)
        vo_bar = vo_bar_vec[0]  # np.zeros_like(wk)
        ws = ws_vec[0]  # 0
        vs = vs_vec[0]  # 0
        vk = vk_vec[0]  # 0
    """


    func_calls, f = wrap_function(f, args)
    if fprime is None:
        #grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
        myfprime = aSNAQ.wrap_function(aSNAQ.gradient_param_shift, (fun, 0, 500))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gnorm = 1
    while (k<maxiter) and  (gnorm > gtol):
        gfk = myfprime(wk + mu * vk)

        if k == 0: F.append(gfk)
        # two loop recursion

        q = gfk
        tau = len(sk_vec)
        a = np.zeros(tau)
        for i in reversed(range(tau)):
            rho = 1 / np.dot(yk_vec[i].T, sk_vec[i])
            a[i] = rho * np.dot(sk_vec[i].T, q)
            q = q - np.dot(a[i], yk_vec[i])
        term = np.sum(np.square(F), 0)
        Hk0 = 1 / np.sqrt(term + eps)
        r = Hk0 * q
        for i in range(tau):
            rho = 1 / np.dot(yk_vec[i].T, sk_vec[i])
            beta = rho * np.dot(yk_vec[i].T, r)
            r = r + sk_vec[i] * (a[i] - beta)
        pk = r
        if vecnorm(pk, 2) == np.inf or vecnorm(pk, 2) == np.nan:
            pk = np.ones_like(wk)

        elif dirNorm:
            pk = pk / vecnorm(pk, 2)  # Exploding gradients (direction normalization)

        if k == 0: F.clear()
        '''
        pk = -gfk
        a = []
        idx = len(sk_vec)
        for i in range(len(sk_vec)):
            a.append(numpy.dot(sk_vec[idx - 1 - i].T, pk) / numpy.dot(sk_vec[idx - 1 - i].T, yk_vec[idx - 1 - i]))
            pk = pk - a[i] * yk_vec[idx - 1 - i]
        term = np.sum(np.square(F), 0)
        Hk0 = 1 / np.sqrt(term + eps)
        pk = Hk0 * pk
        for i in reversed(range(len(sk_vec))):
            b = numpy.dot(yk_vec[idx - 1 - i].T, pk) / numpy.dot(yk_vec[idx - 1 - i].T, sk_vec[idx - 1 - i])
            pk = pk + (a[i] - b) * sk_vec[idx - 1 - i]
        '''

        flag_ret = 1

        vk = mu * vk - alpha_k[0] * pk
        wk = wk + vk

        ws = ws + wk  # +mu*vk
        vs = vs + vk

        gfkp1 = myfprime(wk)
        F.append(gfkp1)
        gnorm = vecnorm(gfkp1,norm)
        if k % L == 0:
            wn_bar = ws / L
            vn_bar = vs / L
            ws = np.zeros_like(wk)
            vs = np.zeros_like(wk)
            if t > 0:
                if f(wn_bar) > gamma * f(wo_bar):
                    sk_vec.clear()
                    yk_vec.clear()
                    mu = np.minimum(mu / mu_fac, mu_clip)
                    mu = np.maximum(mu, mu_init)
                    if clearF: F.clear()
                    # print("Clearing buffers")
                    wk = wo_bar
                    vk = vo_bar
                    flag_ret = 0
                if flag_ret:
                    sk = wn_bar - wo_bar
                    #fisher = np.asarray(F)[:, :, 0].T
                    #yk = np.dot(fisher, np.dot(fisher.T, sk))
                    yk = np.dot(np.asarray(F).T,np.dot(np.asarray(F),sk))
                    mu = np.minimum(mu * mu_fac, mu_clip)
                    # yk = (np.sum(fisher, 1, keepdims=True) * sk) / shape(fisher)[-1]
                    # yk = 0
                    # for i in F:
                    #    yk += np.dot(i,np.dot(i.T,sk))
                    # yk = yk/len(F)
                    if np.dot(sk.T, yk) > eps * np.dot(yk.T, yk):
                        sk_vec.append(sk)
                        yk_vec.append(yk)
                        wo_bar = wn_bar
                        vo_bar = vn_bar
            else:
                wo_bar = wn_bar
                vo_bar = vn_bar
            t += 1
            #t_vec.append(t)

        if callback is not None:
            callback(wk)
        fval = f(wk)
        xk = wk
        print(fval, mu)
        k += 1
        """
        iter.append(k)
        mu_val.append(mu)
        wo_bar_vec.append(wo_bar)  # np.zeros_like(wk)
        vo_bar_vec.append(vo_bar)  # np.zeros_like(wk)
        ws_vec.append(ws)  # 0
        vs_vec.append(vs)  # 0
        vk_vec.append(vk)  # 0
        memL.append(len(sk_vec))
        memF.append(len(F))
        err.append(f(wk))
        """

    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(
        fun=fval,
        jac=gfk,
        #hess_inv=Hk,
        nfev= 0,#sf.nfev,
        njev= 0,#sf.ngev,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=wk,
        nit=k,
    )
    if retall:
        result["allvecs"] = allvecs
    return result
