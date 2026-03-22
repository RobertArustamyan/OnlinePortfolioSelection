"""
BISONS Algorithm for Online Portfolio Selection

Reference: Zimmert, Agarwal, Kale (2022)
"Pushing the Efficiency-Regret Pareto Frontier for Online Learning of
 Portfolios and Quantum States"
Proceedings of Machine Learning Research vol 178:1-45, COLT 2022

Algorithm 1 (page 5), with parameter tuning from Theorem 1 (page 7):
  B = (264/5) * d * log(T)
  eta = 1 / (4B)
  beta = 11 / (7B)

The algorithm also takes epsilon_x and epsilon_u as solver accuracy inputs
(Assumption 1, page 6). These are NOT needed as explicit parameters because
Lemma 3 (page 7, proof in Appendix F page 36) proves that a single damped
Newton step suffices to satisfy Assumption 1 under Theorem 1's tuning:

  For x-solver: ||nabla G_tau(x_{tau+1})||  <=  6*eta
  For u-solver: ||nabla F_hat_tau(u_{tau+1})||  <=  1/(8*sqrt(eta))

  Proof sketch (Appendix F):
    G_tau is sqrt(eta)-self-concordant (Lemma 21).
    One damped Newton step (Lemma 15) gives:
      ||grad_post|| <= 2*M * ||grad_pre||^2
    where M = sqrt(eta). Starting from ||grad_pre|| <= 3*sqrt(eta) (Eq. 13),
      ||grad_post|| <= 2*sqrt(eta) * 9*eta = 18*eta^{3/2} <= 6*eta.  QED.

Regret: O(d^2 * log^2(T)).  Runtime: O(d^3) per step (one matrix inversion).
"""

import numpy as np
from collections import defaultdict


class Bisons:
    def __init__(self, n_stocks, T, B, eta, beta):
        """
        BISONS algorithm for online portfolio selection.

        :param n_stocks: Number of stocks (d in the paper).
        :param T: Time horizon. Not used by the algorithm mechanics — included
                  for logging and interface compatibility. The algorithm's
                  behaviour is fully determined by B, eta, beta.
        :param B: Bias scaling factor (Algorithm 1 input).
                  Scales the linear bias term in the biased surrogate g^e_tau
                  (Eq. 3): g^e_tau(x) = f_hat^e_tau(x) - <x, B*(p^e_tau - p^e_{tau-1})>.
                  Larger B => stronger bias toward recovering underperforming stocks
                  => more conservative. Appears in the reset condition threshold
                  and controls the negative-regret / cost-of-bias tradeoff
                  (Section 4, proof sketch of Theorem 1).
                  Theorem 1 sets B = (264/5) * d * log(T).
        :param eta: Learning rate for the log-barrier regularizer (Algorithm 1 input).
                    Enters as eta^{-1} * R(x) in the FTRL objectives (Eq. 4, 5),
                    where R(x) = -sum_i log(x_i) is the log-barrier regularizer.
                    Smaller eta => stronger regularization => iterates stay closer
                    to the simplex interior (more uniform).
                    Also determines the self-concordance parameter M = sqrt(eta)
                    (Lemma 21: G^e_tau = eta^{-1}*R + quadratic is sqrt(eta)-sc).
                    Theorem 1 sets eta = 1 / (4*B).
        :param beta: Curvature parameter for the quadratic surrogate (Eq. 2):
                     f_hat_t(x) = f_t(x_t) + <x-x_t, grad_t> + (beta/2)*<x-x_t, grad_t>^2.
                     Controls how tightly the surrogate approximates the true loss.
                     Also appears in the reset condition: the epoch resets when any
                     stock's allocation in u is too large relative to 1/(beta*p_i).
                     Theorem 1 sets beta = 11 / (7*B).
        """
        self.n_stocks = n_stocks
        self.d = n_stocks
        self.T = T
        self.B = B
        self.eta = eta
        self.beta = beta

        # ---------------------------------------------------------------------------
        # Self-concordance parameter M (Lemma 21, page 21):
        #
        #   G^e_tau(x) = eta^{-1} * R(x) + Q(x)
        #
        #   where R(x) = -sum log(x_i) is 1-self-concordant (Lemma 20, page 19),
        #   and Q is a quadratic function with PSD Hessian (the accumulated surrogates).
        #
        #   By Lemma 16 (page 16): if f is M-sc, then alpha*f is M/sqrt(alpha)-sc.
        #   So eta^{-1} * R is 1/sqrt(1/eta) = sqrt(eta)-self-concordant.
        #
        #   Adding Q only affects the RHS of the self-concordance definition
        #   (|nabla^3 f[u,u,u]| <= 2M * ||u||^3_{nabla^2 f}), making it easier
        #   to satisfy. So G^e_tau is sqrt(eta)-self-concordant.
        #
        #   This M enters the damped Newton step formula (Lemma 15, page 16):
        #     x+ = x - [nabla^2 f(x)]^{-1} nabla f(x) / (1 + M * lambda(x))
        #   where lambda(x) = ||nabla f(x)||_{[nabla^2 f(x)]^{-1}} is the Newton decrement.
        # ---------------------------------------------------------------------------
        self.M_sc = np.sqrt(self.eta)

        # Uniform initial portfolio (Algorithm 1, "initialize" line):
        #   x^e_1 = u^e_1 = argmin_{x in A} eta^{-1} R(x) = (1/d, ..., 1/d)
        # The minimizer of R(x) = -sum log(x_i) subject to sum(x_i) = 1, x >= 0
        # is the uniform distribution (by symmetry + strict convexity).
        self.p_0 = np.ones(n_stocks) / n_stocks

        # For experiment framework compatibility
        self.prev_portfolio = self.p_0.copy()

        # Global time counter (t in Algorithm 1)
        self.global_t = 0

        # ---------------------------------------------------------------------------
        # Monitoring: history[e][tau] stores per-step diagnostics.
        #
        # For each (epoch e, internal time tau), we record:
        #   global_t       : global time step t
        #   x_played       : x^e_tau, portfolio played this step
        #   u              : u^e_tau, reference solution at time of play
        #   p_bias         : p^e_tau, bias vector at time of play
        #   r_t            : price relatives received
        #   wealth_factor  : <x^e_tau, r_t>, the daily return multiplier
        #   f_t            : f_t(x_t) = -log(<x_t, r_t>), true loss at played point
        #   grad_t         : nabla f_t(x_t) = -r_t / <x_t, r_t>
        #   x_next         : x^e_{tau+1}, iterate after Newton step on G
        #   u_next         : u^e_{tau+1}, iterate after Newton step on F_hat
        #   p_bias_next    : p^e_{tau+1}, bias after update
        #   decrement_x    : Newton decrement for x-solve (lambda in Lemma 15)
        #   decrement_u    : Newton decrement for u-solve
        #   reset          : whether epoch reset was triggered after this step
        # ---------------------------------------------------------------------------
        self.history = defaultdict(dict)

        # Initialize first epoch
        self.epoch = 1
        self.tau = 1
        self._reset_epoch()

    def _reset_epoch(self):
        """
        Reset all epoch-specific state for a new epoch e.

        Called at initialization (e=1) and whenever the reset condition fires.
        After reset, the algorithm forgets all history from previous epochs
        (Algorithm 1: "e <- e+1, tau <- 1").

        Initialization (Algorithm 1, "initialize" line):
          x^e_1 = u^e_1 = argmin_{x in A} eta^{-1}*R(x) = (1/d, ..., 1/d)

          p^e_0 = d * 1_d   (the paper's notation; a d-vector of all d's)
              Since x^e_1 = (1/d, ..., 1/d), we have (x^e_1)^{-1} = (d, ..., d).
              The bias update rule Eq. 6: p^e_{tau,i} = max{p^e_{tau-1,i}, (x^e_{tau,i})^{-1}}.
              Applying at tau=1: p^e_1 = max{p^e_0, (x^e_1)^{-1}} = max{d*1, d*1} = d*1.
              So p^e_0 = p^e_1 = d*1, and the first bias increment is zero.

          G^e_0(x) = F_hat^e_0(x) = eta^{-1} * R(x)  (no losses yet)
              Represented implicitly: we store running sums that start at zero,
              and the gradient/Hessian functions add the barrier term.

        Internal representation of the cumulative functions:

          The surrogate at step s (Eq. 2):
            f_hat_s(x) = f_s(x_s) + <x - x_s, grad_s> + (beta/2)*<x - x_s, grad_s>^2
            where grad_s = nabla f_s(x_s) = -r_s / <x_s, r_s>.

          Its gradient w.r.t. x:
            nabla f_hat_s(x) = grad_s + beta * (grad_s^T (x - x_s)) * grad_s
                             = grad_s * (1 + beta * grad_s^T x - beta * grad_s^T x_s)

          Its Hessian (constant in x):
            nabla^2 f_hat_s = beta * grad_s @ grad_s^T

          The biased surrogate (Eq. 3):
            g^e_s(x) = f_hat^e_s(x) - <x, B * (p^e_s - p^e_{s-1})>

          Cumulative G^e_tau (Eq. 4):
            G^e_tau(x) = sum_{s=1}^{tau} g^e_s(x) + eta^{-1} * R(x)

          Its gradient (using telescoping: sum B*(p_s - p_{s-1}) = B*(p_tau - p_0)):
            nabla G^e_tau(x) = eta^{-1}*(-1/x)                        [barrier]
                              + sum_grads                               [linear from surrogates]
                              + beta * H_surr @ x                       [quadratic from surrogates]
                              - beta * sum_gx                           [constant from surrogates]
                              - B * (p_bias - p_bias_init)              [telescoped bias]

          Cumulative F_hat^e_tau (Eq. 5) is the same without the bias:
            nabla F_hat^e_tau(x) = eta^{-1}*(-1/x) + sum_grads + beta*H_surr@x - beta*sum_gx

          Hessian (shared by G and F_hat, since bias is linear):
            nabla^2 G^e_tau(x) = nabla^2 F_hat^e_tau(x)
                               = eta^{-1} * diag(1/x_i^2) + beta * H_surr
        """
        d = self.d

        # ---- Iterates ----

        # x^e_tau: the portfolio iterate, output of APPROX-SOLVEx(G^e_{tau-1}, x^e_{tau-1}).
        # Played as the actual portfolio at each step.
        self.x_t = np.ones(d) / d

        # u^e_tau: reference iterate, output of APPROX-SOLVEu(F_hat^e_{tau-1}, u^e_{tau-1}).
        # FTRL solution over UNBIASED surrogates. Not played — only used in the
        # reset condition to detect when a stock's allocation exceeds 1/(beta*p_i).
        self.u_t = np.ones(d) / d

        # ---- Bias vector ----

        # p^e_tau: bias vector (Eq. 6), updated as max{p^e_{tau-1,i}, (x^e_{tau,i})^{-1}}.
        # Tracks the largest inverse allocation seen per stock in this epoch.
        # At init: p^e_0 = p^e_1 = d * 1_d (see above).
        self.p_bias = np.ones(d) * d

        # p^e_0: frozen copy of the initial bias for this epoch, needed to compute
        # the telescoped bias sum: B * (p^e_tau - p^e_0).
        self.p_bias_init = np.ones(d) * d

        # ---- Running sums for gradient/Hessian ----

        # sum_{s=1}^{tau} grad_s, where grad_s = -r_s / <x_s, r_s>
        self.sum_grads = np.zeros(d)

        # sum_{s=1}^{tau} grad_s @ grad_s^T  (the Hessian of accumulated surrogates / beta)
        self.H_surr = np.zeros((d, d))

        # sum_{s=1}^{tau} (grad_s^T x_s) * grad_s
        # Note: grad_s^T x_s = (-r_s / <x_s, r_s>)^T x_s = -<r_s, x_s>/<x_s, r_s> = -1.
        # So this simplifies to sum_{s} (-1) * grad_s = -sum_grads.
        # However, we store it separately for clarity and to avoid sign errors.
        # The gradient formulas use (sum_grads - beta * sum_gx), and with
        # sum_gx = -sum_grads this becomes sum_grads + beta * sum_grads = (1+beta)*sum_grads.
        # We keep both for explicit correspondence with the paper's derivation.
        self.sum_gx = np.zeros(d)

    # -------------------------------------------------------------------------
    # Gradient and Hessian evaluation
    # -------------------------------------------------------------------------

    def _hessian(self, x):
        """
        Hessian of G^e_tau(x) or F_hat^e_tau(x) at point x.

        nabla^2 G(x) = eta^{-1} * diag(1/x_i^2)  +  beta * H_surr

        The first term comes from nabla^2 R(x) where R(x) = -sum log(x_i):
          nabla^2 R(x) = diag(1/x_i^2).
        The second term comes from the accumulated surrogate Hessians:
          nabla^2 f_hat_s = beta * grad_s @ grad_s^T.

        Same for both G and F_hat since the bias term is linear (zero Hessian).
        """
        return (1.0 / self.eta) * np.diag(1.0 / (x ** 2)) + self.beta * self.H_surr

    def _grad_G(self, x):
        """
        Gradient of G^e_tau at x (biased cumulative FTRL objective, Eq. 4).

        nabla G^e_tau(x) = eta^{-1} * (-1/x)
                         + sum_grads
                         + beta * H_surr @ x
                         - beta * sum_gx
                         - B * (p_bias - p_bias_init)
        """
        return (
            (1.0 / self.eta) * (-1.0 / x)
            + self.sum_grads
            + self.beta * (self.H_surr @ x)
            - self.beta * self.sum_gx
            - self.B * (self.p_bias - self.p_bias_init)
        )

    def _grad_F(self, x):
        """
        Gradient of F_hat^e_tau at x (unbiased cumulative FTRL objective, Eq. 5).

        nabla F_hat^e_tau(x) = eta^{-1} * (-1/x)
                             + sum_grads
                             + beta * H_surr @ x
                             - beta * sum_gx

        Same as nabla G minus the bias term.
        """
        return (
            (1.0 / self.eta) * (-1.0 / x)
            + self.sum_grads
            + self.beta * (self.H_surr @ x)
            - self.beta * self.sum_gx
        )

    # -------------------------------------------------------------------------
    # Projected damped Newton step (Lemma 15 + simplex constraint)
    # -------------------------------------------------------------------------

    def _projected_hessian_inv(self, H):
        """
        Compute the projected Hessian inverse for the simplex constraint sum(x) = 1.

        The action set A = {x : x >= 0, sum(x) = 1} lies in the affine subspace
        {x : <1, x> = 1}. The tangent space is {v : <1, v> = 0}.

        From Appendix B.2 (page 19), for functions restricted to the trace-1
        constraint (which reduces to sum-1 for diagonal matrices):

          [nabla^2_restricted]^{-1} = H^{-1} - (H^{-1} @ 1 @ 1^T @ H^{-1}) / (1^T @ H^{-1} @ 1)

        This is the Schur complement formula for the constrained inverse.
        It satisfies [nabla^2_restricted]^{-1} @ 1 = 0, so any Newton direction
        d_N = -[nabla^2_restricted]^{-1} @ grad has sum(d_N) = 0, preserving
        the simplex equality constraint.

        In the paper's notation (page 19):
          lim_{lambda->inf} [nabla^2 H_lambda(X)]^{-1}
            = [nabla^2 H(X)]^{-1} - ([nabla^2 H]^{-1} Id Id^T [nabla^2 H]^{-1})
                                     / (Id^T [nabla^2 H]^{-1} Id)
        For diagonal matrices (portfolio case), Id -> 1_d.
        """
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.inv(H + 1e-8 * np.eye(self.d))

        ones = np.ones(self.d)
        H_inv_ones = H_inv @ ones
        denom = ones @ H_inv_ones

        return H_inv - np.outer(H_inv_ones, H_inv_ones) / denom

    def _damped_newton_step(self, x, grad_fn):
        """
        One damped Newton step for a sqrt(eta)-self-concordant function
        restricted to the probability simplex.

        Lemma 15 (page 16): For an M-self-concordant function f, the damped
        Newton step is:

          x+ = x - [nabla^2 f(x)]^{-1} nabla f(x) / (1 + M * lambda(x))

        where lambda(x) = ||nabla f(x)||_{[nabla^2 f(x)]^{-1}} is the
        Newton decrement. The damping factor (1 + M*lambda) ensures that
        x+ stays in the domain, and Lemma 15 guarantees:

          lambda(x+) <= 2M * lambda(x)^2

        For the simplex-constrained case, we replace H^{-1} with the
        projected Hessian inverse (see _projected_hessian_inv), so the
        Newton direction d_N = -H_pi_inv @ grad satisfies sum(d_N) = 0.

        :param x: Current iterate, strictly in simplex interior.
        :param grad_fn: Callable x -> gradient vector.
        :return: (x_new, newton_decrement)
        """
        g = grad_fn(x)
        H = self._hessian(x)
        H_pi_inv = self._projected_hessian_inv(H)

        # Newton direction in the tangent space of the simplex
        d_N = -H_pi_inv @ g

        # Newton decrement: lambda = sqrt(g^T H_pi_inv g)
        # Equivalently: lambda = ||d_N||_{H} = sqrt(d_N^T H d_N)
        # We use the gradient form which is equivalent.
        decrement_sq = g @ H_pi_inv @ g
        decrement = np.sqrt(max(decrement_sq, 0.0))

        # Damped step (Lemma 15)
        damping = 1.0 + self.M_sc * decrement
        x_new = x + d_N / damping

        # Numerical safety: the log-barrier guarantees x stays interior in exact
        # arithmetic, but floating point may produce tiny negatives.
        x_new = np.maximum(x_new, 1e-15)
        x_new /= x_new.sum()

        return x_new, decrement

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def get_portfolio(self):
        """
        Return the current portfolio x^e_tau to be played.

        The log-barrier regularization R(x) = -sum log(x_i) keeps all iterates strictly in the simplex interior.
        """
        return self.x_t.copy()

    def update(self, r_t, p_used):
        """
        Update algorithm state after observing price relatives r_t.

        Implements one iteration of Algorithm 1's for-loop (page 5):

          1. Receive f_t from playing x_t = x^e_tau  (we receive r_t, compute grad_t)
          2. Construct surrogate f_hat^e_tau and accumulate into G^e_tau, F_hat^e_tau
          3. x^e_{tau+1} <- APPROX-SOLVEx(G^e_tau, x^e_tau)   [one damped Newton step]
          4. u^e_{tau+1} <- APPROX-SOLVEu(F_hat^e_tau, u^e_tau)  [one damped Newton step]
          5. p^e_{tau+1,i} = max{p^e_{tau,i}, (x^e_{tau+1,i})^{-1}}       (Eq. 6)
          6. Check reset: if exists i: 2(1+6*eta)*beta * u_{tau+1,i} >= (p_{tau+1,i})^{-1}
             Equivalently: u_{tau+1,i} * p_{tau+1,i} >= 1 / (2*(1+6*eta)*beta)

        :param r_t: Price relatives vector (d-dimensional, non-negative).
        :param p_used: Portfolio that was actually played (= x^e_tau from get_portfolio).
        """
        x_t = p_used
        e = self.epoch
        tau = self.tau
        self.global_t += 1

        # --- Step 1: Compute loss gradient at the played portfolio ---
        #
        # f_t(x) = -log(<x, r_t>),  so nabla f_t(x_t) = -r_t / <x_t, r_t>
        wealth_factor = np.dot(x_t, r_t)

        if wealth_factor <= 0:
            print(f"Warning: Non-positive wealth factor {wealth_factor}, skipping update")
            return

        grad_t = -r_t / wealth_factor
        f_t_value = -np.log(wealth_factor)

        # --- Step 2: Accumulate surrogate quantities ---
        #
        # These three running sums encode the cumulative surrogate gradient/Hessian.
        # After this step, they represent tau losses (including the current one).
        self.sum_grads += grad_t
        self.H_surr += np.outer(grad_t, grad_t)
        self.sum_gx += np.dot(grad_t, x_t) * grad_t

        # --- Step 3: Damped Newton step for x^e_{tau+1} ---
        #
        # Approximately minimises G^e_tau(x) = sum g^e_s(x) + eta^{-1}*R(x)
        # starting from x^e_tau. The gradient _grad_G uses the current p_bias
        # (= p^e_tau, computed at the end of the previous step or at init),
        # which is correct: the bias in G^e_tau is B*(p^e_tau - p^e_0).
        x_next, decrement_x = self._damped_newton_step(self.x_t, self._grad_G)

        # --- Step 4: Damped Newton step for u^e_{tau+1} ---
        #
        # Approximately minimises F_hat^e_tau(x) = sum f_hat^e_s(x) + eta^{-1}*R(x)
        # (no bias term) starting from u^e_tau.
        u_next, decrement_u = self._damped_newton_step(self.u_t, self._grad_F)

        # --- Step 5: Update bias (Eq. 6) ---
        #
        # p^e_{tau+1,i} = max{p^e_{tau,i}, (x^e_{tau+1,i})^{-1}}
        #
        # This tracks the worst (smallest) allocation x_i seen in this epoch.
        # If x_{tau+1,i} is smaller than any previous allocation, the inverse
        # is larger, so p_i increases. The bias g^e_tau penalises the FTRL
        # objective to give more weight to stocks that performed poorly.
        p_next = np.maximum(self.p_bias, 1.0 / x_next)

        # --- Step 6: Reset condition ---
        #
        # Algorithm 1: "if exists i: (2(1+6*eta)*beta) * u^e_{tau+1,i} >= (p^e_{tau+1,i})^{-1}"
        #
        # Rearranging: u_{tau+1,i} >= 1 / (2*(1+6*eta)*beta * p_{tau+1,i})
        # Equivalently: u_{tau+1,i} * p_{tau+1,i} >= 1 / (2*(1+6*eta)*beta)
        #
        # Interpretation (Section 4, page 8): the reset fires when the unbiased
        # FTRL solution u allocates too much to some stock relative to that stock's
        # worst historical inverse allocation. This means the stock has recovered
        # beyond what the current epoch's bias can compensate. The key insight is
        # that when reset fires, the "negative regret" term -<u, p_tau - p_0>*B
        # is large enough (Lemma 39) to cancel all positive regret in the completed
        # epoch, making the epoch's total regret non-positive.
        reset_threshold = 1.0 / (2.0 * (1.0 + 6.0 * self.eta) * self.beta)
        reset_triggered = bool(np.any(u_next * p_next >= reset_threshold))

        # --- Record history ---
        self.history[e][tau] = {
            'global_t': self.global_t,
            'x_played': x_t.copy(),
            'u': self.u_t.copy(),
            'p_bias': self.p_bias.copy(),
            'r_t': r_t.copy(),
            'wealth_factor': wealth_factor,
            'f_t': f_t_value,
            'grad_t': grad_t.copy(),
            'x_next': x_next.copy(),
            'u_next': u_next.copy(),
            'p_bias_next': p_next.copy(),
            'decrement_x': decrement_x,
            'decrement_u': decrement_u,
            'reset': reset_triggered,
        }

        # --- Apply updates ---
        self.x_t = x_next
        self.u_t = u_next
        self.p_bias = p_next

        if reset_triggered:
            self.epoch += 1
            self.tau = 1
            self._reset_epoch()
        else:
            self.tau += 1

        # Experiment framework compatibility
        self.prev_portfolio = p_used.copy()

    def simulate_trading(self, price_relatives_sequence, stock_prices_sequence=None,
                         verbose=True, verbose_days=100):
        """
        Simulate trading over a sequence of price relatives.

        :param price_relatives_sequence: Array of shape (n_days, d) with price relatives.
        :param stock_prices_sequence: Ignored (interface compatibility with cost-aware models).
        :param verbose: Print progress.
        :param verbose_days: Print frequency.
        :return: Dictionary with trading results and diagnostics.
        """
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []
        turnovers = []
        num_no_trades = 0

        for day, r_t in enumerate(price_relatives_sequence):
            portfolio = self.get_portfolio()
            portfolios_used.append(portfolio.copy())

            turnover = np.sum(np.abs(portfolio - self.prev_portfolio))
            turnovers.append(turnover)
            if turnover < 1e-10:
                num_no_trades += 1

            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            self.update(r_t, portfolio)

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, "
                      f"Epoch = {self.epoch}, Tau = {self.tau}, "
                      f"Turnover = {turnover:.6f}, "
                      f"Portfolio = [{', '.join([f'{x:.4f}' for x in portfolio])}]")

        n_days = len(price_relatives_sequence)
        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'num_days': n_days,
            'total_epochs': self.epoch,
            'history': dict(self.history),
            # Fields required by BaseOptunaExperiment.run_single_training
            'transaction_costs': np.zeros(n_days),
            'total_transaction_cost': 0.0,
            'turnovers': np.array(turnovers),
            'avg_turnover': float(np.mean(turnovers)),
            'max_turnover': float(np.max(turnovers)) if turnovers else 0.0,
            'trade_frequency': 1 - (num_no_trades / n_days) if n_days > 0 else 0.0,
        }