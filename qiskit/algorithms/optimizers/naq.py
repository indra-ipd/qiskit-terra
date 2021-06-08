from typing import Optional

import numpy as np
from numpy import Inf
from scipy.optimize import OptimizeResult
from .scipy_optimizer import SciPyOptimizer
from scipy.optimize.optimize import (
    _prepare_scalar_function,
    #_check_unknown_options,
    #vecnorm,
    _status_message,
    _line_search_wolfe12,
    _LineSearchError,
)


class NAQ(SciPyOptimizer):
    """
    NAQ

    See https://www.jstage.jst.go.jp/article/nolta/8/4/8_289/_pdf
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "mu", "lineSearch", "analytical_grad","reduce_fev"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: Optional[int] = None,
        maxfev: int = 1024,
        disp: bool = False,
        mu: float = 0.9,
        lineSearch: str = "armijo",
        analytical_grad: bool = True,
        reduce_fev: bool = False,
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
        super().__init__(method=naq, options=options, **kwargs)

def vecnorm(x, ord=2):
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)

# pylint: disable=invalid-name
def naq(
    fun,
    x0,
    args=(),
    jac=None,
    callback=None,
    mu=0.9,
    global_conv=True,
    gtol=1e-5,
    norm=Inf,
    eps=1e-8,
    maxiter=None,
    lineSearch="armijo",
    reduce_fev = False,
    disp=False,
    return_all=False,
    finite_diff_rel_step=None,
    gamma=1e-5,
    analytical_grad=False,
    **unknown_options,
):
    """
    Minimization of scalar function of one or more variables using the
    NAQ algorithm.

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
    reduce_fev : bool, default: False, if True uses MoQ implementation to reduce the number of function evaluations

    """
    #_check_unknown_options(unknown_options)
    retall = return_all

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0)


    if analytical_grad:
        f = fun
        myfprime = NAQ.wrap_function(NAQ.gradient_param_shift, (fun, 0, 500))
        sf = _prepare_scalar_function(
            f, x0, myfprime, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step
         )
        f = sf.fun
        myfprime = sf.grad
    else:
        jac = NAQ.wrap_function(NAQ.gradient_num_diff,(fun, eps, 500))#sf.grad

        #sf = _prepare_scalar_function(
        #    fun, x0, myfprime, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step
        #)
        sf = _prepare_scalar_function(
            fun, x0, jac, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step
        )
        f = sf.fun
        myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)
    if reduce_fev:
        curr_grad = np.copy(gfk)
        prev_grad = np.copy(gfk)
    err = []
    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError) as e:
            raise ValueError(
                "The user-provided " "objective function must " "return a scalar value."
            ) from e

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    vk = np.zeros_like(x0)
    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    if mu == "adaptive":
        theta_k = 1
    else:
        muVal = mu
    while (gnorm > gtol) and (k < maxiter):
        if mu == "adaptive":
            theta_kp1 = (
                (gamma - (theta_k * theta_k))
                + np.sqrt(
                    ((gamma - (theta_k * theta_k)) * (gamma - (theta_k * theta_k)))
                    + 4 * theta_k * theta_k
                )
            ) / 2
            muVal = np.minimum((theta_k * (1 - theta_k)) / (theta_k * theta_k + theta_kp1), 0.95)
            theta_k = theta_kp1

        xmuv = xk + muVal * vk
        if k > 0:
            if reduce_fev:
                gfk = (1 + muVal) * curr_grad - muVal * prev_grad
            else:
                gfk = myfprime(xmuv)

        pk = -np.dot(Hk, gfk)

        pknorm = vecnorm(pk, ord=norm)
        if pknorm > 1000:
            delta = 1e-7
        else:
            delta = 1e-4

        try:
            if type(lineSearch) != str:
                alpha_k = lineSearch

            elif lineSearch == "armijo":
                # Armijo Line Search
                alpha_k = 1
                old_old_fval = f(xmuv)
                warnflag = 2
                while alpha_k > 1e-4:
                    old_fval = f(xmuv + alpha_k * pk)
                    RHS = old_old_fval + 1e-3 * alpha_k * np.dot(gfk.T, pk)
                    if old_fval <= RHS:
                        warnflag = 0
                        break
                    else:
                        alpha_k *= 0.5
                if warnflag:
                    break

            elif lineSearch == "wolfe":
                alpha_k = 1
                old_old_fval = f(xmuv)
                old_fval = f(xmuv + alpha_k * pk)

                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                    f, myfprime, xmuv, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100
                )

            elif lineSearch == "explicit":
                LHS = f(xmuv + pk)
                RHS = f(xmuv) + 1e-3 * np.dot(gfk.T, pk)
                if LHS <= RHS:
                    alpha_k = 1
                else:
                    # first iter
                    if k == 0:
                        L = 100
                        old_old_fval = LHS + np.linalg.norm(gfk) / 2
                        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                            f, myfprime, xmuv, pk, gfk, LHS, old_old_fval, amin=1e-100, amax=1e100
                        )

                    else:
                        L = 100 * (vecnorm(yk, ord=norm) / vecnorm(sk, ord=norm))
                        Qk = L * np.eye(N)
                        pkQ = np.sqrt(np.dot(pk.T, np.dot(Qk, pk)))
                        alpha_k = -(delta * np.dot(gfk.T, pk)) / np.square(pkQ)

        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        vkp1 = muVal * vk + alpha_k * pk
        xkp1 = xk + vkp1
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - (xk + muVal * vk)
        xk = xkp1
        vk = vkp1

        gfkp1 = myfprime(xkp1)
        if reduce_fev:
            prev_grad = np.copy(curr_grad)
            curr_grad = np.copy(gfkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1

        # global convergence
        if global_conv:
            p_times_q = np.dot(sk.T, yk)
            if gnorm > 1e-2:
                const = 2.0
            else:
                const = 100.0
            if p_times_q < 0:
                p_times_p = np.dot(sk.T, sk)
                zeta = const - (p_times_q / (p_times_p * gnorm))
            else:
                zeta = const

            yk = yk + zeta * gnorm * sk

        if callback is not None:
            callback(xk)

        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if gnorm <= gtol:
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.0:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1.0 / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])

    fval = old_fval

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
        hess_inv=Hk,
        nfev=sf.nfev,
        njev=sf.ngev,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xk,
        nit=k,
    )
    if retall:
        result["allvecs"] = allvecs
    return result
