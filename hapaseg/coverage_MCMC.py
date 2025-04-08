import enum
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import sortedcontainers as sc
import sys
from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB
import warnings
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .model_optimizers import (
    PoissonRegression,
    CovLNP_NR_prior,
    covLNP_ll_prior,
)

# turn off warnings for statsmodels fitting
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", HessianInversionWarning)
np.seterr(divide="ignore")

colors = mpl.cm.get_cmap("tab10").colors


WINDOW_SIZES = 10 * 2 ** np.arange(7)
"""The base list of kernel sizes to use when searching for double breakpoints."""


class MoveType(enum.Enum):
    JOIN = enum.auto()
    SPLIT = enum.auto()


class AllelicCluster:
    observations: np.ndarray
    """A [N] np-array containing the observed coverage of each bin."""
    covariates: np.ndarray
    """A [N, covariates] np-array containing the covariates at each bin."""
    mu: np.floating
    """The log-mean of the log-normal Poisson distribution."""
    beta: np.ndarray
    """The coefficient of each covariate."""
    lsigma: float
    """The log-sigma parameter of the log-normal Poisson distribution."""
    sigma: float
    """The exponent of the sigma parameter of the log-normal Poisson distribution."""
    bin_exposure: int
    """A constant factor shifting all the expected bin coverages, equal to the bin width."""
    segment_ID: np.ndarray
    """An [N] np-array containing the index of the segment of each observation."""
    segments: sc.SortedSet
    """A sorted list containing the start index of each segment."""
    segment_lens: sc.SortedDict
    """A sorted dict mapping the segment index to the length of the segment."""
    F: sc.SortedList
    """
    A sorted list containing the segment breakpoints.
    Breakpoints are kept twice, once as segment end and once as segment start.
    """

    cache_LL: Dict[Tuple[int, int], float]
    """A cache for segment log-likelihoods."""
    cache_mu_lsigma_hess: Dict[Tuple[int, int], Tuple[float, float, np.ndarray]]
    """A cache for segment LNP log-mean, log-sigma, and hessian."""

    def __init__(
        self,
        r: np.ndarray,
        C: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        bin_width=1,
    ):
        # cluster wide params
        self.observations = r
        self.covariates = C
        assert len(mu) == 1
        self.mu = mu.item()
        self.beta = beta
        self.lsigma = 1  # log sigma lnp parameter
        self.sigma = np.exp(self.lsigma)  # sigma lnp parameter

        # bin width is set if were are dealing with uniform bin lengths
        # we use this as an exposure to account for the lack of an explicit
        # covaraiate
        self.bin_exposure = bin_width

        # individual segment lnp params
        self.mu_i = 0
        self.sigma_i = 0
        self.lsigma_i = 0
        self.exp = 0

        # all segment params
        self.lsigma_i_arr = np.ones(len(self.observations))
        self.mu_i_arr = np.zeros(len(self.observations))

        self.lamda = 1e-10
        self.alpha_prior = 1e-5
        self.beta_prior = 4e-3

        # keep cache of previously computed breakpoints for fast splitting
        self.cache_LL = {}
        self.cache_mu_lsigma_hess = {}

        # all intervals start out in one segment, which also means no breakpoints
        self.segment_ID = np.zeros(len(self.observations))
        self.segments = sc.SortedSet([0])
        self.segment_lens = sc.SortedDict([(0, len(self.observations))])

        # keep cache of previously computed breakpoints for fast splitting
        # these breakpoints keys are in the form (st, en, breakpoint)
        self.breakpoint_cache = {}

        self.phase_history = []
        self.F = sc.SortedList()
        self.F.update([0, len(self.observations)])

        self.ll_traces = []

        self.opt_views = []
        self.mui_views = []
        self.lsigma_views = []

        self._init_params()

    # fits lnp model to this cluster using initial params as starting point
    def _init_params(self):
        self.mu, self.lgsigma, _ = self.lnp_init()
        self.lgsigma_i_arr = np.ones(len(self.observations)) * self.lsigma
        self.sigma_i_arr = np.exp(self.lsigma_i_arr)

    def get_seg_ind(self, seg: int) -> Tuple[int, int]:
        """Gets the start and end positions of a semgent.

        Args:
            seg (int): The start position of a segment.
        """
        return seg, seg + self.segment_lens[seg]

    def get_join_seg_ind(self, seg_l, seg_r):
        return seg_l, seg_r + self.segment_lens[seg_r]

    def get_ll(self):
        bdy = np.r_[list(self.segments), len(self.observations)]
        bdy = np.c_[bdy[:-1], bdy[1:]]
        ll = 0
        for st, en in bdy:
            # lookup in cache
            if (curr_ll := self.cache_LL.get((st, en), None)) is not None:
                ll += curr_ll
            else:
                ll += self.ll_segment(
                    (st, en), self.mu_i_arr[st], self.lsigma_i_arr[st]
                )
        return ll

    # read in the merged cluster assignments from burnin scatter jobs and
    # fill in data structures for cluster mcmc accordingly
    def _init_burnin(self, reconciled_assignments):
        if len(self.observations) != len(reconciled_assignments):
            raise ValueError(
                "reconciled_assignemnts did not match the number of cluster bins, expected {} got {}".format(
                    len(self.observations), len(reconciled_assignments)
                )
            )

        self.segment_ID = np.ones(len(reconciled_assignments), dtype=int) * -1
        self.segments = sc.SortedSet()
        self.segment_lens = sc.SortedDict()
        self.F = sc.SortedList([0, len(self.segment_ID)])

        prev = 0
        prev_index = 0
        for i, v in enumerate(reconciled_assignments):
            # if we notice a value change we add the segment to the datastructures
            if v != prev:
                self.segment_ID[prev_index:i] = prev_index
                self.segments.add(prev_index)
                self.segment_lens[prev_index] = i - prev_index
                self.F.update([i, i])

                prev = v
                prev_index = i
        # once we reach the end we add the last segment
        self.segment_ID[prev_index:] = prev_index
        self.segments.add(prev_index)
        self.segment_lens[prev_index] = i - prev_index

        # refit poisson to get new beta
        pois_regr = PoissonRegression(
            np.exp(
                np.log(self.observations).flatten()
                - np.log(self.bin_exposure)
                - self.mu_i_arr
            )[:, None],
            self.covariates,
            np.ones(self.observations.shape).reshape(-1, 1),
        )
        mu, beta = pois_regr.fit()
        self.beta = beta

        # get new lnp values
        self._init_params()

        # now we need to update the lsigma and mu_i arrays based on the burnin segmentation
        for row in np.array(self.F).reshape(-1, 2):
            mu_i, lsigma_i, _ = self.lnp_optimizer(row)
            self.mu_i_arr[row[0] : row[1]] = mu_i
            self.lsigma_i_arr[row[0] : row[1]] = lsigma_i

        print("finished reading in burnin data")

    def lnp_init(self) -> Tuple[np.floating, np.floating, np.ndarray]:
        # fit LNP
        lnp = CovLNP_NR_prior(
            self.observations,
            self.beta,
            self.covariates,
            exposure=np.log(self.bin_exposure),
            init_prior=False,
            lamda=self.lamda,
            mu_prior=self.mu,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
        )
        return lnp.fit()

    def lnp_optimizer(
        self,
        ind: Tuple[int, int],
    ) -> Tuple[float, float, np.ndarray]:
        """Computes the LNP parameters for the given index.

        Args:
            ind (tuple[int, int]): A tuple containing the start and stop indices of a segment.

        Returns:
            Tuple: A tuple containing the log-mean, log-sigma, and hessian of the log-normal Poisson distribution fit on the target range.
        """
        if ind in self.cache_mu_lsigma_hess:
            return self.cache_mu_lsigma_hess[ind]

        # cache miss; compute values
        lnp = CovLNP_NR_prior(
            self.observations[ind[0] : ind[1]],
            self.beta,
            self.covariates[ind[0] : ind[1]],
            exposure=np.log(self.bin_exposure),
            mu_prior=self.mu,
            lamda=self.lamda,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
            init_prior=False,
        )

        try:
            res = lnp.fit()
        except ValueError:
            print(f"AllelicCluster::lnp_optimize({ind}) did not converge!")
            print(f"obs={self.observations[ind[0] : ind[1]]}")
            print(f"beta={self.beta}")
            print(f"exposure={self.bin_exposure}")
            print(f"lambda={self.lamda}")
            print(f"alpha_prior={self.alpha_prior}")
            print(f"beta_prior={self.beta_prior}")
            # try adding some jitter
            try:
                lnp = CovLNP_NR_prior(
                    self.observations[ind[0] : ind[1]],
                    self.beta,
                    self.covariates[ind[0] : ind[1]],
                    exposure=np.log(self.bin_exposure),
                    mu_prior=self.mu,
                    lamda=self.lamda,
                    alpha_prior=self.alpha_prior,
                    beta_prior=self.beta_prior,
                    extra_roots=True,
                    init_prior=False,
                )
                res = lnp.fit()
            except ValueError:
                res = (np.nan, np.nan, np.full((2, 2), np.nan))

        # save to cache
        self.cache_mu_lsigma_hess[ind] = res
        return res

    # method for refitting beta value of the cluster
    def refit_beta(self):
        # fit poisson to get new beta
        pois_regr = PoissonRegression(
            np.exp(
                np.log(self.observations).flatten()
                - np.log(self.bin_exposure)
                - self.mu_i_arr
            ).reshape(-1, 1),
            self.covariates,
            np.ones(self.observations.shape).reshape(-1, 1),
        )

        mu, beta = pois_regr.fit()
        self.beta = beta

        # get new lnp mu and lsigma values
        self._init_params()

        self.cache_LL.clear()
        self.cache_mu_lsigma_hess.clear()

        # refit to segments to get lsigma_i and mu_i arrays
        for row in np.array(self.F).reshape(-1, 2):
            mu_i, lsigma_i, _ = self.lnp_optimizer(row)
            self.mu_i_arr[row[0] : row[1]] = mu_i
            self.lsigma_i_arr[row[0] : row[1]] = lsigma_i

    ## caluculating overall ll of allelic cluster under lnp model
    def ll_segment(
        self,
        ind: Tuple[int, int],
        mu_i: Union[np.ndarray, float],
        lgsigma: Union[np.ndarray, float],
    ):
        exposure = np.log(self.bin_exposure)
        mu_tot = mu_i
        ll = covLNP_ll_prior(
            self.observations[ind[0] : ind[1]],
            mu_tot,
            lgsigma,
            self.covariates[ind[0] : ind[1]],
            self.beta,
            exposure=exposure,
            mu_prior=self.mu,
            lamda=self.lamda,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
        )
        return ll

    # method for fast difference kernel calculation
    def _run_kernel(self, residuals: np.ndarray, window: int) -> np.ndarray:
        """
        Computes the difference kernel on the residuals.
        The kernel is `[-1]*window + [1]*window`, and computes the absolute difference between the `window` positions and the `window` following positions.
        With `window==1`, this would be equivalent to `np.diff`.

        Args:
            residuals (np.ndarray): An `[N]` np-array containing the observations.
            window (int): Half the size of the kernel/

        Returns:
            np.ndarray: An `[N - 2*window + 1]` np-array, containing the differences between the sums of two adjecant subsegments.
        """
        kernel = np.concatenate([-np.ones(window), np.ones(window)])
        return np.abs(np.convolve(residuals, kernel, mode="valid"))

    def _change_kernel(
        self, ind: Tuple[int, int], residuals: np.ndarray, window: int
    ) -> List[int]:
        """
        method for convolving a change kernel with a given window across an array of residuals. This change kernel returns
        the absolute difference between the means of the residuals within windows on either side of a rolling change point.

        returns a list of peaks above the 98th quantile that are seperated by at least max(25, window/2)
        """
        difs = self._run_kernel(residuals, window)
        if window > 10:
            x = np.r_[: len(difs)]
            prior = np.exp(2 * np.abs(x - len(difs) / 2) / len(difs))
            difs = prior * difs

        cutoff = np.quantile(difs, 0.98)

        ind_len = ind[1] - ind[0]
        if ind_len > 50000:
            distance = 10000
        elif ind_len > 5000:
            distance = 1000
        else:
            distance = 50
        # use helper function to find peaks above cutoff that are seperated
        peaks, _ = find_peaks(difs, height=cutoff, distance=distance)
        peaks = peaks + ind[0] + window
        return list(peaks)

    def _find_difs(
        self, ind: Tuple[int, int], window_sizes: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        method for narrowing search space of possible split positions in a segment. Finds the indices of the top ~2% of
        the difference kernel values for multiple window sizes in addition to the first and last window bins. Falls back on
        an exhaustive search, which means returning all possible split points.

        returns list of possible split points to try
        """
        residuals = np.exp(
            np.log(self.observations[ind[0] : ind[1]].flatten())
            - (self.mu)
            - (self.covariates[ind[0] : ind[1]] @ self.beta).flatten()
        )

        minimal = False
        ind_len = ind[1] - ind[0]
        if window_sizes is None:
            if ind_len > 50000:
                window_sizes = np.array([1000])
            elif ind_len > 10000:
                window_sizes = np.array([500])
            elif ind_len > 5000:
                window_sizes = np.array([500, 100, 25])
            elif ind_len > 1000:
                window_sizes = np.array([100, 50, 25])
            else:
                minimal = True
                window_sizes = np.array([50, 10])

        window_sizes = window_sizes[window_sizes * 2 < ind_len]

        difs = []
        for window in window_sizes:
            difs.append(self._change_kernel(ind, residuals, window))
        difs_idxs = set().union(*difs)
        difs_idxs = np.r_[list(difs_idxs)]
        # if there are no that pass the threshold return them all
        if len(difs_idxs) == 0:
            print("no significant bins", flush=True)
            return list(np.r_[ind[0] + 2 : ind[1] - 1])

        # add on first and last windows edges if were in a small interval
        # since these are not considered by the kernel
        if minimal:
            difs_idxs = np.r_[difs_idxs, ind[0] + 10, ind[1] - 10]
        return list(difs_idxs)

    def _calculate_splits(
        self, ind: Tuple[int, int], split_indices: List[int]
    ) -> Tuple[
        List[float],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[np.ndarray, np.ndarray]],
    ]:
        """
        Given the indices of the segment and the proposed split indices to try, this method computes the likelihood of each
        split proposal and saves the associated mu, lsigma and Hessian results associated to the split segments.

        Returns a tuple containing four lists:
        * The list of split log-likelihoods.
        * The list of the left and right LNP log-means.
        * The list of the left and right LNP log-sigmas.
        * The list of the left and right Hessians.
        """
        lls = []
        mus = []
        lsigmas = []
        Hs = []
        for ix in split_indices:
            mu_l, lsigma_l, H_l = self.lnp_optimizer((ind[0], ix))
            mu_r, lsigma_r, H_r = self.lnp_optimizer((ix, ind[1]))

            mus.append((mu_l, mu_r))
            lsigmas.append((lsigma_l, lsigma_r))
            Hs.append((H_l, H_r))

            # lookup likelihoods in cache
            # left:
            if (ind[0], ix) in self.cache_LL:
                ll_l = self.cache_LL[ind[0], ix]
            else:
                ll_l = self.ll_segment((ind[0], ix), mu_l, lsigma_l)
                self.cache_LL[ind[0], ix] = ll_l

            # right:
            if (ix, ind[1]) in self.cache_LL:
                ll_r = self.cache_LL[ix, ind[1]]
            else:
                ll_r = self.ll_segment((ix, ind[1]), mu_r, lsigma_r)
                self.cache_LL[ix, ind[1]] = ll_r

            lls.append(ll_l + ll_r)

        return lls, mus, lsigmas, Hs

    def _detailed_sampling(self, ind, lls, split_indices, mus, lsigmas, Hs):
        """
        This method is used for refining the search space of possible splits, which is important for
        both finding the best split and for properly normalizing the split likelihoods. It works by iteratively sampling
        points in either direction of a proposed split until it reaches a potential split point which is less than 1/1k
        as likely as the most likely point seen so far.

        returns an amended list of likelihoods and optimal parameters for the original split indicies in addition to the
        ones sampled.
        """
        if ind[1] - ind[0] > 25000:
            max_samples = 5
        else:
            max_samples = 15
        # find the most likely index and those indices within a significant range
        # (rn 5 ll points -> at least 1/150th as likely)
        top_lik = max(lls)
        sig_threshold = top_lik - 5
        within_range = np.r_[lls] > sig_threshold
        sig_idx = np.r_[split_indices][within_range]
        sig_set = set(sig_idx)
        num_samps = 0
        for ix in sig_idx:
            # dont sample areound the split index
            if ix < 0:
                continue
            # sample to the left of the position until were out of the significant range
            ix_l = ix - 1
            r_samples = 0
            while r_samples < max_samples:
                if ix_l < ind[0] + 2:
                    break
                ll_add, mu_add, lsigma_add, Hs_add = self._calculate_splits(
                    ind, [ix_l]
                )
                if ll_add[0] > sig_threshold and ix_l not in sig_set:
                    split_indices.append(ix_l)
                    lls.extend(ll_add)
                    mus.extend(mu_add)
                    lsigmas.extend(lsigma_add)
                    Hs.extend(Hs_add)

                    sig_set.add(ix_l)
                    ix_l -= 1
                    num_samps += 1
                    r_samples += 1
                else:
                    break

            # now do the same for the right side
            ix_r = ix + 1
            l_samples = 0
            while l_samples < max_samples:
                if ix_r >= ind[1] - 1:
                    break
                ll_add, mu_add, lsigma_add, Hs_add = self._calculate_splits(
                    ind, [ix_r]
                )
                if ll_add[0] > sig_threshold and ix_r not in sig_set:
                    split_indices.append(ix_r)
                    lls.extend(ll_add)
                    mus.extend(mu_add)
                    lsigmas.extend(lsigma_add)
                    Hs.extend(Hs_add)

                    sig_set.add(ix_r)
                    ix_r += 1
                    num_samps += 1
                    l_samples += 1
                else:
                    break
        return lls, split_indices, mus, lsigmas, Hs

    def _lls_to_MLs(
        self, lls: List[float], hessians: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        function for calculating the log marginal likelihood of a split given the log likelihood and the hessians for
        the NB fit of each split segment
        """
        MLs = np.zeros(len(lls))
        for i, (ll, hessian) in enumerate(zip(lls, hessians)):
            laplacian = self._get_log_ML_gaussint_split(hessian[0], hessian[1])
            # the split results in a nan make it impossible to split there
            if np.isnan(laplacian):
                laplacian = -1e50
            MLs[i] = ll + laplacian
        return MLs

    def _get_split_likelihoods(self, ind: Tuple[int, int], debug=False):
        """
        Function for computing log MLs for all possible split points of interest. If a segment is small enough we will
        consider every possible split point, otherwise will use change kernel and detailed sampling
        """
        ind_len = ind[1] - ind[0]
        # do not allow segments to be split into segments of size less than 2
        if ind_len < 4 and not debug:
            raise Exception(
                "segment was too small to split: length: ", ind[1] - ind[0]
            )

        # if the cluster is sufficiently small we interrogate the whole space of splits
        if ind_len < 150:
            split_indices = list(np.r_[ind[0] + 2 : ind[1] - 1])
            lls, mus, lsigmas, Hs = self._calculate_splits(ind, split_indices)
        # otherwise we deploy our change kernel to limit our search space
        else:
            split_indices = self._find_difs(ind)

            # get lls from the difference kernel guesses
            lls, mus, lsigmas, Hs = self._calculate_splits(ind, split_indices)
            lls, split_indices, mus, lsigmas, Hs = self._detailed_sampling(
                ind, lls, split_indices, mus, lsigmas, Hs
            )
        MLs = self._lls_to_MLs(lls, Hs)

        return split_indices, MLs, mus, lsigmas

    # computes ML component from hessian approximation for a single segment
    def _get_log_ML_gaussint_join(self, Hess):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-Hess))) / 2

    # computes ML component from hessian approximation for two split segments
    def _get_log_ML_gaussint_split(self, H1, H2):
        return (
            2 * np.log(2 * np.pi)
            - (np.log(np.linalg.det(-H1) * np.linalg.det(-H2))) / 2
        )

    # computes the log ML of joining two segments
    def _log_ML_join(self, ind):
        mu_share, lsigma_share, H_share = self.lnp_optimizer(ind)

        # lookup cache
        if ind in self.cache_LL:
            ll_join = self.cache_LL[ind]
        else:
            ll_join = self.ll_segment(ind, mu_share, lsigma_share)
            self.cache_LL[ind] = ll_join
        return (
            mu_share,
            lsigma_share,
            self._get_log_ML_gaussint_join(H_share) + ll_join,
        )

    def split(self, debug: bool) -> int:
        """
        Split segment method. This method chooses a segment at random
        from the allelic cluster, and calculates the MLs of splitting the segment either every possible position or positions
        informed by the change kernel heuristic. Compares the MLs of splitting the segment at each position with the ML of
        the joint cluster and probabilistically takes an action proportional to likelihood.

        Returns -1 if splitting was skipped due to segment being too small to split (<4 bins), 0 if the choice was to
        leave the cluster as is, and <breakpoint> if the segment was split at <breakpoint>
        """
        # pick a random segment
        seg = np.random.choice(self.segments)

        # if segment is a singleton then skip
        ## if segment is less than 4 bins then skip
        if self.segment_lens[seg] < 4:
            return -1

        ind = self.get_seg_ind(seg)

        split_indices, log_split_MLs, all_mus, all_lsigmas = (
            self._get_split_likelihoods(ind, debug=debug)
        )
        _, _, log_join_ML = self._log_ML_join(ind)

        log_MLs = np.r_[log_split_MLs, log_join_ML]
        split_indices.append(-1)

        max_ML = max(log_MLs)
        k_probs = np.exp(log_MLs - max_ML) / np.exp(log_MLs - max_ML).sum()

        if np.isnan(k_probs).any():
            print(
                "skipping split iteration due to nan. log MLs: ",
                log_MLs,
                flush=True,
            )
            return 0
        choice_idx = np.random.choice(len(split_indices), p=k_probs)
        break_idx = split_indices[choice_idx]
        if break_idx > 0:
            # split the segments
            mus = all_mus[choice_idx]
            lsigmas = all_lsigmas[choice_idx]
            # update segment assignments for new segment
            self.segment_ID[break_idx : ind[1]] = break_idx
            self.segments.add(break_idx)
            self.segment_lens[break_idx] = ind[1] - break_idx

            # update old seglen
            self.segment_lens[ind[0]] = break_idx - ind[0]

            # update mu_i and lsigma_i values
            self.mu_i_arr[ind[0] : break_idx] = mus[0]
            self.mu_i_arr[break_idx : ind[1]] = mus[1]
            self.lsigma_i_arr[ind[0] : break_idx] = lsigmas[0]
            self.lsigma_i_arr[break_idx : ind[1]] = lsigmas[1]

            self.F.update([break_idx, break_idx])
            self.phase_history.append(self.F.copy())
            return break_idx
        # otherwise we have chosen to join and we do nothing
        return 0

    def join(self, debug) -> int:
        """
        Join segments method. This method chooses a segment at random from the allelic cluster, and probabilistically
        chooses to join that segment with its neighbor to the right. This action is taken with probability equal to the
        ratio of the joint and split MLs

        returns -1 if there is only one segment, 0 if the proposed join is rejected, and <breakpoint> if a join occurred,
        where <breakpoint> is the left most index of the right segment joined.
        """
        num_segs = len(self.segments)
        # if theres only one segment, skip
        if num_segs == 1:
            return -1

        # otherwise pick a left segment to join
        seg_l_ind = np.random.choice(num_segs - 1)
        seg_l = self.segments[seg_l_ind]
        seg_r = self.segments[seg_l_ind + 1]
        ind = self.get_join_seg_ind(seg_l, seg_r)

        lls_split, _, _, Hs = self._calculate_splits(ind, [seg_r])
        log_split_ML = lls_split[0] + self._get_log_ML_gaussint_split(
            Hs[0][0], Hs[0][1]
        )

        mu_share, lsigma_share, log_join_ML = self._log_ML_join(ind)

        log_MLs = np.r_[log_split_ML, log_join_ML]
        k_probs = np.exp(log_MLs - np.logaddexp.reduce(log_MLs))

        if np.isnan(k_probs).any():
            print(
                "skipping iter due to nan in join. log_MLs:",
                log_MLs,
                flush=True,
            )
            return 0
        join_choice = np.random.choice([0, 1], p=k_probs)

        if join_choice:
            # join the segments
            # update segment assignments for new segment
            self.segment_ID[ind[0] : ind[1]] = seg_l
            self.segments.discard(seg_r)
            del self.segment_lens[seg_r]
            self.segment_lens[seg_l] = ind[1] - ind[0]

            # update mu_i and lsigma_i values
            self.mu_i_arr[ind[0] : ind[1]] = mu_share
            self.lsigma_i_arr[ind[0] : ind[1]] = lsigma_share

            # we need to discard twice since the breakpoint is saved
            # both as an end to an interval and a start to the next interval
            self.F.discard(seg_r)
            self.F.discard(seg_r)
            self.phase_history.append(self.F.copy())
            return seg_r

        return 0

    def get_mu_lsigma_hess(
        self, ind: Tuple[int, int], cache=False
    ) -> Tuple[float, float, np.ndarray]:
        if ind in self.cache_mu_lsigma_hess:
            return self.cache_mu_lsigma_hess[ind]

        mu, lsigma, hess = self.lnp_optimizer(ind)
        if cache:
            self.cache_mu_lsigma_hess[ind] = (mu, lsigma, hess)

        return mu, lsigma, hess

    def get_segment_ml(self, ind: Tuple[int, int], cache=True) -> float:
        """
        Gets the maximum-likelihood of the segment.

        Args:
            ind (Tuple[int, int]): A tuple containing the start point and end point of the segment.
            cache (bool, optional): Whether to cache the LNP intermediates. Defaults to False.
        """
        # Getting the Hessian of the segment.
        mu, lsigma, hess = self.get_mu_lsigma_hess(ind, cache=cache)

        # Getting the maximal likelihood.
        if ind in self.cache_LL:
            ll = self.cache_LL[ind]
        else:
            ll = self.ll_segment(ind, mu, lsigma)
            if cache:
                self.cache_LL[ind] = ll

        return ll + self._get_log_ML_gaussint_join(hess)

    def get_breakpoint_pairs(
        self, ind: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """
        Gets all pairs of breakpoints for the double-split action in the given segment.

        Args:
            ind (Tuple[int, int]): The identifier for a single segment.
        """

        residuals = np.exp(
            np.log(self.observations[ind[0] : ind[1]].flatten())
            - (self.mu)
            - (self.covariates[ind[0] : ind[1]] @ self.beta).flatten()
        )

        breakpoint_pairs = set()
        for kernel_size in WINDOW_SIZES[WINDOW_SIZES * 2 < ind[1] - ind[0]]:
            split_indices = sorted(
                self._change_kernel(ind, residuals, kernel_size)
            )

            for start, stop in zip(split_indices, split_indices[1:]):
                breakpoint_pairs.add((start, stop))

        return breakpoint_pairs

    def split_double(self, debug: bool) -> int:
        """
        A double-split method. This method chooses a segment at random and attempts to excise from it a sub-segment, or takes a sub-segment and attempts to remove it.

        To reduce the size of the search space, the sub-segment is chosen using the change kernel heuristic, and among all possible sub-segments
        one is chosen with the probability equal to the log-likelihood.

        Returns 0 if the choice was to leave the cluster as is, and <breakpoint> if the segment was split at <breakpoint>
        """

        # Pick a random segment
        # We use a Gibbs sampler formulation, wherein we pick a segment and pick the next state with probabilites corresponding to the likelihood.
        # To allow reverse moves, if two adjacent breakpoints are a breakpoint pair of the segment containing them, then we also allow the reverse move.
        reverse_move_segments: Dict[int, Tuple[int, int]] = {}
        for idx, seg in enumerate(self.segments):
            if (idx == 0) or (idx == len(self.segments) - 1):
                continue
            nxt = seg + self.segment_lens[seg]
            prev: int = self.segments[idx - 1]
            nxt_nxt = nxt + self.segment_lens[nxt]

            if (seg, nxt) in self.get_breakpoint_pairs((prev, nxt_nxt)):
                reverse_move_segments[seg] = (prev, nxt_nxt)
        # Choosing a random move.
        if np.random.random() < len(self.segments) / (
            len(self.segments) + len(reverse_move_segments)
        ):
            # Making a split move.
            seg = np.random.choice(self.segments)
            ind = self.get_seg_ind(seg)
        else:
            seg: int = np.random.choice(list(reverse_move_segments))
            small_ind = self.get_seg_ind(seg)
            ind = reverse_move_segments[seg]

            # We start by removing the segment seg to make the join and split actions similar.
            self.segment_ID[ind[0] : ind[1]] = ind[0]
            self.segments.discard(small_ind[0])
            self.segments.discard(small_ind[1])
            del self.segment_lens[small_ind[0]]
            del self.segment_lens[small_ind[1]]
            self.segment_lens[ind[0]] = ind[1] - ind[0]

            # update mu_i and lsigma_i values

            mu_share, lsigma_share, _ = self.get_mu_lsigma_hess(ind)

            self.mu_i_arr[ind[0] : ind[1]] = mu_share
            self.lsigma_i_arr[ind[0] : ind[1]] = lsigma_share

            # we need to discard twice since the breakpoint is saved
            # both as an end to an interval and a start to the next interval
            self.F.discard(small_ind[0])
            self.F.discard(small_ind[0])
            self.F.discard(small_ind[1])
            self.F.discard(small_ind[1])

        # Making sure we don't count windows twice
        breakpoint_pairs = list(self.get_breakpoint_pairs(ind))
        if len(breakpoint_pairs) == 0:
            return -1

        base_ml = self.get_segment_ml(ind)
        pair_mls = [
            self.get_segment_ml((ind[0], bp[0]))
            + self.get_segment_ml((bp[0], bp[1]))
            + self.get_segment_ml((bp[1], ind[1]))
            for bp in breakpoint_pairs
        ] + [base_ml]

        # Choosing a split.
        mls = np.array(pair_mls)
        split_probs: np.ndarray = np.exp(mls - np.max(mls))
        split_probs /= np.sum(split_probs)

        if np.isnan(split_probs).any():
            print(
                "skipping split iteration due to nan. log MLs: ",
                mls,
                flush=True,
            )
            return 0

        split = np.random.choice(np.arange(len(split_probs)), p=split_probs)
        # Choosing not to split.
        if split == len(breakpoint_pairs):
            return 0

        low_break, high_break = breakpoint_pairs[split]

        self.segments.add(low_break)
        self.segments.add(high_break)

        self.F.update([low_break, low_break, high_break, high_break])

        self.segment_ID[low_break:high_break] = low_break
        self.segment_ID[high_break : ind[1]] = high_break

        self.segment_lens[ind[0]] = low_break - ind[0]
        self.segment_lens[low_break] = high_break - low_break
        self.segment_lens[high_break] = ind[1] - high_break

        self.phase_history.append(self.F.copy())

        low_mu, low_ls, _ = self.get_mu_lsigma_hess(
            (ind[0], low_break), cache=True
        )
        self.mu_i_arr[ind[0] : low_break] = low_mu
        self.lsigma_i_arr[ind[0] : low_break] = low_ls

        mid_mu, mid_ls, _ = self.get_mu_lsigma_hess(
            (low_break, high_break), cache=True
        )
        self.mu_i_arr[low_break:high_break] = mid_mu
        self.lsigma_i_arr[low_break:high_break] = mid_ls

        high_mu, high_ls, _ = self.get_mu_lsigma_hess(
            (high_break, ind[1]), cache=True
        )
        self.mu_i_arr[low_break:high_break] = high_mu
        self.lsigma_i_arr[low_break:high_break] = high_ls
        return low_break


class Coverage_MCMC_AllClusters:
    """
    This class is for running segmentation on all allelic clusters on the same node.
    Each iteration first randomly chooses a cluster, and then proposes and split/join operation.
    """

    def __init__(self, n_iter, r, C, Pi, bin_width=1):
        self.n_iter = n_iter
        self.r = r
        self.C = C
        self.Pi = Pi
        self.bin_exposure = bin_width

        # for now assume that the Pi vector assigns each bin to exactly one cluster
        self.c_assignments = np.argmax(self.Pi, axis=1)
        self.n_clusters = Pi.shape[1]
        self.clusters = [None] * self.n_clusters
        self.cluster_sizes = np.array(
            [np.sum(self.c_assignments == i) for i in range(self.n_clusters)]
        )
        self.cluster_probs = np.ones(self.n_clusters) / self.n_clusters

        self.burnt_in = False

        self.num_segments = np.ones(self.n_clusters)

        self.mu_i_samples = []
        self.lsigma_i_samples = []
        self.F_samples = []

        self.ll_clusters = np.zeros(self.n_clusters)
        self.ll_iter = []

        # first we find good starting values for mu and beta for each cluster by fitting a poisson model
        pois_regr = PoissonRegression(self.r, self.C, self.Pi)
        mu_0, self.beta = pois_regr.fit()

        for k in range(self.n_clusters):
            cluster_mask = self.c_assignments == k
            new_acluster = AllelicCluster(
                self.r[cluster_mask],
                self.C[cluster_mask, :],
                mu_0[k],
                self.beta,
            )
            self.clusters[k] = new_acluster

            # set initial ll
            self.ll_clusters[k] = new_acluster.get_ll()

    def save_sample(self):
        """
        Adds the current MCMC state to the list of saved MCMC states.
        """
        mu_i_save = []
        lsigma_i_save = []
        F_save = []

        for clust in self.clusters:
            mu_i_save.append(clust.mu_i_arr.copy())
            lsigma_i_save.append(clust.lsigma_i_arr.copy())
            F_save.append(clust.F.copy())
        self.mu_i_samples.append(mu_i_save)
        self.lsigma_i_samples.append(lsigma_i_save)
        self.F_samples.append(F_save)

    def pick_cluster(self, n_it: int) -> int:
        """
        Randomly picks a cluster from the current set of clusters.

        In the first 1000 iterations, the cluster is picked uniformly from the set of clusters.
        After that, the cluster is picked with probability proportional to its length.

        Args:
            n_it (int): The number of MCMC iterations done up to now.

        Returns:
            int: The index of a random cluster.
        """
        # randomly select clusters with equal probabilites for the first 1k iterations then select based on size
        # TODO: tweak this to dynamically decide when to switch selection probabilites
        if n_it < 1000:
            return np.random.choice(range(self.n_clusters))
        return np.random.choice(range(self.n_clusters), p=self.cluster_probs)

    def run(self, debug=False, stop_after_burnin=False):
        print("starting MCMC coverage segmentation...")

        past_it = 0

        # for n_it in tqdm.tqdm(range(n_iter)):
        n_it = 0
        while self.n_iter > len(self.F_samples):
            # check if we have burnt in
            if n_it > 500 and not self.burnt_in and not n_it % 100:
                if np.diff(np.array(self.ll_iter[-500:])).mean() < 0:
                    print("burnt in!")
                    self.burnt_in = True
                    past_it = n_it
                    if stop_after_burnin:
                        print(
                            "Burn-in complete: ll: {} n_it: {}".format(
                                self.ll_clusters.sum(), n_it
                            )
                        )
                        return

            # update cluster probs to reflect number of segments in each cluster
            if n_it > 1000 and not n_it % 100:
                self.cluster_probs = self.num_segments / self.num_segments.sum()

            # save dynamicaly thinned chain samples
            if (
                not n_it % 50
                and self.burnt_in
                and n_it - past_it > self.num_segments.sum()
            ):
                self.save_sample()
                past_it = n_it

            # decide whether to split or join on this iteration
            cluster_pick = self.pick_cluster(n_it)
            # cluster_pick = 1
            if np.random.rand() > 0.5:
                # split
                res = self.clusters[cluster_pick].split(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_clusters[cluster_pick] = self.clusters[
                        cluster_pick
                    ].get_ll()
                    self.num_segments[cluster_pick] += 1
            else:
                # join
                res = self.clusters[cluster_pick].join(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_clusters[cluster_pick] = self.clusters[
                        cluster_pick
                    ].get_ll()
                    self.num_segments[cluster_pick] -= 1
            n_it += 1
            self.ll_iter.append(self.ll_clusters.sum())

    def update_beta(self, total_exposure):
        endog = self.r.flatten()
        exog = self.C
        exposure = np.ones(self.r.shape) * self.bin_exposure
        start_params = np.r_[self.beta.flatten(), 1]
        sNB = statsNB(
            endog,
            exog,
            exposure=exposure,
            offset=total_exposure,
            start_params=start_params,
        )
        res = sNB.fit(start_params=start_params, disp=0)
        return res.params[:-1]

    def prepare_results(self):
        # convert saved Cov_MCMC cluster results into global result arrays
        seg_samples = np.zeros((self.Pi.shape[0], len(self.F_samples)))
        mu_i_samples = np.zeros((self.Pi.shape[0], len(self.F_samples)))
        mu_global = np.zeros(self.Pi.shape[0])

        pi_argmax = self.Pi.argmax(1)
        for it in range(len(self.F_samples)):
            global_seg_counter = 0
            for c in range(len(self.F_samples[it])):
                og_positions = np.where(pi_argmax == c)[0]
                if it == 0:
                    mu_global[og_positions] = self.clusters[c].mu
                seg_intervals = np.array(self.F_samples[it][c]).reshape(-1, 2)
                mu_i_samples[og_positions, it] = self.mu_i_samples[it][c]
                for st, en in seg_intervals:
                    seg_ind = og_positions[st:en]
                    seg_samples[seg_ind, it] = global_seg_counter
                    global_seg_counter += 1

        overall_exposure = mu_global + mu_i_samples[:, -1]
        global_beta = self.update_beta(overall_exposure)
        return seg_samples, global_beta, mu_i_samples

        # TODO: add visualization script

    """
	This class is for running segmentation on a single allelic clusters, which allows for scattering cluster segmentations
	across nodes. Under the current structuring of the package the global beta values still need to be calculated in each
	node, hence the need for passing the full r, C, and Pi matrices to each call. In the future this should be pushed to
	the initialization sequence in the run_coverage_mcmc class.
	"""


class Coverage_MCMC_SingleCluster:
    n_iter: int
    observations: np.ndarray
    """A [N, 1] np-array containing the coverage data."""
    covariates: np.ndarray
    """A [N, covariates] np-array containing for each bin the covariates at that bin."""
    beta: np.ndarray
    """A [covariates, 1] np-array containing the coefficient of each covariate."""
    mu: np.ndarray
    """A [clusters, 1] np-array containing the coverage mean of each cluster. (According to the Poisson model?)"""
    bin_width: int
    """The width of all bins."""
    burnt_in: bool
    """Whether the MCMC has finished the burn-in."""
    from_burnin: bool
    """Whether the MCMC was initialized at a burned in position."""
    num_segments: int
    """The current segment count of the MCMC."""
    cluster: AllelicCluster
    """The object holding the MCMC state and actually performing the MCMC moves."""

    ll_cluster: float
    """The current state log-likelihood."""

    def __init__(
        self,
        n_iter: int,
        observations: np.ndarray,
        covariates: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        bin_width=1,
    ):
        self.n_iter = n_iter
        self.observations = observations
        self.covariates = covariates
        self.beta = beta
        self.mu = mu
        self.bin_width = bin_width
        # for now assume that the Pi vector assigns each bin to exactly one cluster

        self.burnt_in = False

        # flag designating whether we initialized from burnin scatter
        self.from_burnin = False

        self.num_segments = 1

        self.mu_i_samples = []
        self.lsigma_i_samples = []
        self.F_samples = []
        self.ll_samples = []

        self.ll_cluster = 0
        self.ll_iter = []

        self.cluster = AllelicCluster(
            self.observations,
            self.covariates,
            self.mu,
            self.beta,
            bin_width=self.bin_width,
        )

        # set initial ll
        self.ll_cluster = self.cluster.get_ll()

    # allow user to pass in reconciled burnin cluster assignments
    def init_burnin(self, reconciled_assignments):
        self.from_burnin = True
        self.cluster._init_burnin(reconciled_assignments)
        # reset initial ll
        self.ll_cluster = self.cluster.get_ll()

        # set num segments
        self.num_segments = len(self.cluster.segments)

    def save_sample(self):
        self.mu_i_samples.append(self.cluster.mu_i_arr.copy())
        self.lsigma_i_samples.append(self.cluster.lsigma_i_arr.copy())
        self.F_samples.append(self.cluster.F.copy())
        self.ll_samples.append(self.cluster.get_ll())

    def run(self, debug=False, stop_after_burnin=False):
        """Runs the MCMC.

        The MCMC has two kinds of moves: Cluster splits and cluster joins, both handled by the `AllelicCluster` object.
        The MCMC is considered burned in when the log-likelihood is smaller than the one 50 steps before it.

        Args:
            debug (bool, optional): A deprecated flag, causing an exception to be thrown by some split moves. Defaults to False.
            stop_after_burnin (bool, optional): Whether to stop the MCMC after the burn-in, or continue for some more iterations. Defaults to False.
        """

        print(
            "Starting MCMC coverage segmentation ...",
            flush=True,
            file=sys.stderr,
        )

        past_it = 0
        n_it = 0
        if self.from_burnin:
            min_it = max(200, self.num_segments)
            refit = True if len(self.observations) > 1000 else False
        else:
            min_it = min(200, max(50, self.observations.shape[0]))
            refit = False

        lookback_len = min(50, min_it)

        while self.n_iter > len(self.F_samples):
            # status update
            if not n_it % 50:
                print("n_it: {}".format(n_it), flush=True)
                if not self.burnt_in and refit:
                    self.cluster.refit_beta()

            # check if we have burnt in
            if n_it >= min_it and not self.burnt_in and not n_it % 50:
                if np.diff(np.array(self.ll_iter[-lookback_len:])).mean() <= 0:
                    print("burnt in!", flush=True)
                    self.burnt_in = True
                    past_it = n_it
                    if stop_after_burnin:
                        print(
                            "Burn-in complete: ll: {} n_it: {}".format(
                                self.ll_cluster, n_it
                            )
                        )
                        return

            # save dynamicaly thinned chain samples
            if (
                not n_it % 25
                and self.burnt_in
                and n_it - past_it > self.num_segments
            ):
                self.save_sample()
                past_it = n_it

            action = np.random.randint(3)
            if action == 0:
                # split
                res = self.cluster.split(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_cluster = self.cluster.get_ll()
                    self.num_segments += 1
            elif action == 1:
                # join
                res = self.cluster.join(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_cluster = self.cluster.get_ll()
                    self.num_segments -= 1
            elif action == 2:
                res = self.cluster.split_double(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_cluster = self.cluster.get_ll()
                    self.num_segments += 1
            else:
                raise ValueError(
                    f"The action is sampled in [0, 4), and should not be {action}!"
                )
            n_it += 1
            self.ll_iter.append(self.ll_cluster)

    # return just the local beta in this case since we cant do global calculation until we see all of the clusters
    def prepare_results(self):
        num_draws = len(self.F_samples)
        num_bins = len(self.cluster.observations)

        segmentation_samples = np.zeros((num_bins, num_draws))
        mu_i_full = np.zeros((num_bins, num_draws))

        for d in range(num_draws):
            seg_counter = 0
            seg_intervals = np.array(self.F_samples[d]).reshape(-1, 2)
            mu_i_full[:, d] = self.mu_i_samples[d]

            for st, en in seg_intervals:
                segmentation_samples[st:en, d] = seg_counter
                seg_counter += 1

        return segmentation_samples, self.beta, mu_i_full, self.ll_samples

    def visualize_cluster_samples(self, savepath):
        residuals = np.exp(
            np.log(self.cluster.observations.flatten())
            - (self.cluster.mu.flatten())
            - np.log(self.bin_width)
            - (self.cluster.covariates @ self.cluster.beta).flatten()
        )
        num_draws = len(self.F_samples)
        fig, axs = plt.subplots(
            num_draws,
            figsize=(8, num_draws * 2),
            sharex=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
        )
        if isinstance(axs, np.ndarray):
            ax_lst = axs.flatten()
        else:
            ax_lst = [axs]
        for d in range(num_draws):
            if d % 2:
                # drop yaxis ticks for every other axis since they overlap
                axs[d].set_yticks([])
            ax_lst[d].scatter(np.r_[: len(residuals)], residuals, s=1)
            ax_lst[d].set_xlim([0, len(residuals)])
            ax_lst[d].set_ylim([residuals.min() - 0.05, residuals.max() + 0.05])

            hist = np.array(self.F_samples[d]).reshape(-1, 2)
            for j, r in enumerate(hist):
                ax_lst[d].add_patch(
                    mpl.patches.Rectangle(
                        (r[0], 0),
                        r[1] - r[0],
                        residuals.max(),
                        fill=True,
                        alpha=0.3,
                        color=colors[j % 10],
                    )
                )
        axs[0].set_title("Segment Coverage MCMC Samples")
        fig.text(
            0.0,
            0.5,
            "corrected coverage residuals",
            va="center",
            rotation="vertical",
        )
        axs[-1].set_xlabel("bin_position")
        fig.tight_layout()
        plt.savefig(savepath, dpi=150)
