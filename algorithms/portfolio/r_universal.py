from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import brentq
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

class RUniversalPortfolio:
    def __init__(self, n_stocks, epsilon=None, eta=None, T=None, A=None,
                 m=None, S=None, delta=None, delta_0=None, use_damping=True,
                 n_processes=None, use_parallel=False, verbose=False):
        '''
        R-UNIVERSAL Portfolio Algorithm

        :param n_stocks: Number of stocks
        :param epsilon: Performance approximation parameter
        :param eta: Probability of failure
        :param T: Number of trading days
        :param A: Constant from paper
        :param m: Number of samples per day
        :param S: Number of random walk steps per sample
        :param delta: Grid spacing
        :param delta_0: Minimum coordinate value
        :param use_damping: If True, use Q_t (with damping); if False, use P_t
        :param n_processes: number of parallel processes
        :param use_parallel: Enable multiprocessing
        :param verbose: For portfolio/wealth tracing
        '''

        self.n = n_stocks
        self.use_damping = use_damping
        self.use_parallel = use_parallel
        self.n_processes = n_processes if n_processes else cpu_count()
        self.t = 0
        self.current_portfolio = np.ones(n_stocks) / n_stocks
        self.price_relatives_history = []

        self.verbose = True

        # Find mode (manual or theoretical)
        manual_params_provided = all(x is not None for x in [m, S, delta, delta_0])
        theoretical_params_provided = all(x is not None for x in [epsilon, eta, T])

        if manual_params_provided:
            self.mode = "manual"
            self.m = m
            self.S = S
            self.delta = delta
            self.delta_0 = delta_0
            # Keeping values (can be None)
            self.epsilon = epsilon
            self.eta = eta
            self.T = T
            self.A = A

        elif theoretical_params_provided:
            self.mode = "theoretical"
            self.epsilon = epsilon
            self.eta = eta
            self.T = T
            self.A = A
            self._calculate_theoretical_params()

        else:
            raise ValueError(
                "Must provide either:\n"
                "  - Theoretical params: epsilon, eta, T\n"
                "  - Manual params: m, S, delta, delta_0"
            )

        self._print_config()

    def _calculate_theoretical_params(self):
        # Calculate parameters according to Theorem 2
        n = self.n
        T = self.T
        epsilon = self.epsilon
        eta = self.eta
        A = self.A

        # delta_0 <= epsilon / (8*n*T*(n+T)^2)
        self.delta_0 = epsilon / (8 * n * T * (n + T) ** 2)

        # Solve: delta * log(1/delta) = epsilon*delta_0 / (A*(n+T)^2)
        target_value = (epsilon * self.delta_0) / (A * (n + T) ** 2)
        self.delta = self._solve_for_delta(target_value)

        # m >= 64*T^2*(n+T)*ln(nT/eta) / epsilon^2
        self.m = 64 * T ** 2 * (n + T) * np.log(n * T / eta) / epsilon ** 2

        # S >= A*n/delta^2 * log((n+T)/(epsilon*delta))
        if self.delta > 0:
            self.S = A * n / self.delta ** 2 * np.log((n + T) / (epsilon * self.delta))
        else:
            self.S = 10000

    @staticmethod
    def _solve_for_delta(target_value):
        """
        Solve: delta * log10(1/delta) = target_value
        Using scipy's Brent method
        """
        if target_value <= 0:
            return 1e-6

        max_value = 1.0 / np.e * np.log10(np.e)  # approx. 0.1597
        if target_value > max_value:
            return 1.0 / np.e

        if target_value < 1e-15:
            return 0.01

        def equation_to_solve(delta):
            if delta <= 0:
                return float('inf')
            return delta * np.log10(1.0 / delta) - target_value

        try:
            solution = brentq(equation_to_solve, 1e-10, 0.368, xtol=1e-15)
            print('in solution')
            check = solution * np.log10(1.0 / solution)
            if abs(check - target_value) / target_value > 0.1:
                return min(0.01, np.sqrt(target_value * 10))
            return solution
        except:
            return min(0.01, np.sqrt(target_value * 10))

    def _print_config(self):
        print(f"R-UNIVERSAL Portfolio Configuration")
        print(f"Mode: {self.mode}")
        print(f"Stocks: {self.n}")
        print(f"Damping: {'ENABLED (Q_t)' if self.use_damping else 'DISABLED (P_t)'}")
        print(f"Parallel: {'Enabled({} processes)'.format(self.n_processes) if self.use_parallel else 'Disabled'}")

        if self.mode == "theoretical":
            print(f"\nTheoretical Parameters:")
            print(f" epsilon = {self.epsilon}")
            print(f" eta = {self.eta}")
            print(f" T = {self.T}")
            print(f" A = {self.A}")

        print(f"\nAlgorithm Parameters:")
        print(f" delta_0 = {self.delta_0:.6e}")
        print(f" delta = {self.delta:.6e}")
        print(f" m = {self.m}")
        print(f" S = {self.S}")

    def P_t(self, portfolio):
        """
        Cumulative wealth function P_t(b)
        P_t(b) = MUL(i=1 to t) b*x_i
        """
        if len(self.price_relatives_history) == 0:
            return 1.0

        wealth = 1.0
        for price_relatives in self.price_relatives_history:
            daily_return = np.dot(portfolio, price_relatives)
            wealth *= daily_return

        return wealth

    def Q_t(self, portfolio):
        """
        Damped wealth function Q_t(b)
        Q_t(b) = P_t(b) * min(exp((b_n - 2*delta_0)/(n*delta)), 1)

        Used when use_damping=True
        """
        P_t_value = self.P_t(portfolio)

        if not self.use_damping:
            return P_t_value

        # damping factor
        b_n = portfolio[-1]
        damping_exp = (b_n - 2 * self.delta_0) / (self.n * self.delta)
        damping_factor = min(np.exp(damping_exp), 1.0)

        return P_t_value * damping_factor

    def _wealth_function(self, portfolio):
        # Choose between P_t and Q_t based on use_damping
        return self.Q_t(portfolio) if self.use_damping else self.P_t(portfolio)

    def _random_walk_step(self, r):
        """
        One step of random walk

        1. use_damping=True: Pick j from {0,...,n-2}, pair with last coord
        2. use_damping=False: Pick any two different coordinates
        """
        coords = np.random.choice(self.n, size=2, replace=False)
        i, j = coords[0], coords[1]

        if self.use_damping:
            i = -1

        X = np.random.choice([-1, 1])
        new_r = r.copy()
        new_r[j] += X * self.delta
        new_r[i] -= X * self.delta

        # Check constraints
        if new_r[j] >= self.delta_0 and new_r[i] >= self.delta_0:
            wealth_current = self._wealth_function(r)
            wealth_proposed = self._wealth_function(new_r)

            if wealth_current > 0 and wealth_proposed > 0:
                acceptance_prob = min(1.0, wealth_proposed / wealth_current)
                if np.random.random() < acceptance_prob:
                    return new_r

        return r

    def _sample_portfolio(self):
        """Generate one portfolio sample via S steps of random walk"""
        # Start at uniform distribution
        r = np.ones(self.n) / self.n

        # Snap to grid
        r = np.round(r / self.delta) * self.delta
        r = np.maximum(r, self.delta_0)
        r = r / np.sum(r)  # Renormalize

        # Perform S steps of random walk
        for _ in range(self.S):
            r = self._random_walk_step(r)

        return r

    def _sample_portfolio_wrapper(self, _):
        """Used for multiprocessing"""
        return self._sample_portfolio()

    def update(self, price_relatives):
        self.price_relatives_history.append(price_relatives.copy())
        self.t += 1

        # Generate m samples (parallel or sequential)
        if self.use_parallel:
            with Pool(processes=self.n_processes) as pool:
                samples = pool.map(self._sample_portfolio_wrapper, range(self.m))
            samples = np.array(samples)
        else:
            samples = np.array([self._sample_portfolio() for _ in range(self.m)])

        # Average samples
        self.current_portfolio = np.mean(samples, axis=0)
        self.current_portfolio /= np.sum(self.current_portfolio)

        if self.verbose:
            wealths = np.array([self._wealth_function(sample) for sample in samples])

            for i, (sample, wealth) in enumerate(zip(samples, wealths)):
                print(f"Sample {i + 1}: Portfolio = {sample}, Wealth = {wealth:.6f}")

            print("Portfolio mean: ", self.current_portfolio)

        return self.current_portfolio.copy()

    def get_portfolio(self):
        return self.current_portfolio.copy()

    def simulate_trading(self, price_relatives_sequence, verbose=True):
        """
        Simulate trading over a sequence of days
        """
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []

        for day, price_relatives in enumerate(price_relatives_sequence):
            portfolio = self.get_portfolio()
            portfolios_used.append(portfolio.copy())

            daily_return = np.dot(portfolio, price_relatives)
            wealth *= daily_return
            daily_wealth.append(wealth)

            self.update(price_relatives)

            if verbose and ((day + 1) % 10 == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}: Portfolio = {portfolio}")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'num_days': len(price_relatives_sequence)
        }
