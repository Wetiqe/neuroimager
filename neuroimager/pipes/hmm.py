import numpy as np
import pandas as pd
from scipy.io import loadmat

from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting


class HmmParser(object):
    def __init__(
        self,
        hmm: str or dict,
        volumes: int,
        subj_num: int,
        sessions=1,
        vpath=None,
        gamma=None,
        xi=None,
        subj_labels=None,
        roi_labels=None,
        auto_parse=True,
        generate_report=True,
    ):
        if isinstance(hmm, str):
            # most simple way to use this class
            self._load_hmm(hmm)
        elif isinstance(hmm, dict):
            # If you already loaded the hmm .mat file
            self.hmm = self.convert_hmm(hmm)
            self.vpath = vpath
            self.gamma = gamma
            self.Xi = xi
        else:
            raise TypeError("hmm must be a file path or a dictionary")
        self.volumes = volumes
        self.subj_num = subj_num
        self.sessions = sessions
        self.roi_labels = roi_labels
        if subj_labels:
            self.subj_labels = subj_labels
        else:
            self.subj_labels = [f"subj{i}" for i in range(subj_num)]
        if self.sessions > 1:
            self.subj_labels = self._generate_subj_labels_with_session()

        self._check_hmm()
        self.K = int(self.hmm["K"])
        self.states_info = {}
        if auto_parse:
            self.vpath_chronnectome = self.parse_vpath_chronnectome()
            self.gamma_chronnectome = self.parse_gamma_chronnectome()
            for i, (mean, conn) in enumerate(zip(self.get_means(), self.get_conns())):
                self.states_info[f"state{i}"] = {"mean": mean, "conn": conn}
        if generate_report:
            self._mean_fig = self.plot_means()
            self._conn_fig = self.plot_conns()

    # prepare the object
    def _check_hmm(self):
        # check the all of the above parameters, if one of them is not provided, raise error
        param_dict = {
            "hmm": self.hmm,
            "vpath": self.vpath,
            "gamma": self.gamma,
            "volumes": self.volumes,
            "subj_num": self.subj_num,
            "sessions": self.sessions,
        }
        flat = []
        for key, item in param_dict.items():
            if item is None:
                flat.append(key)
        if len(flat) > 0:
            raise ValueError(f"The following parameters are missing: {flat}")
        if self.subj_num * self.sessions * self.volumes != self.vpath.shape[0]:
            raise ValueError(
                f"The number of volumes in vpath ({self.vpath.shape[0]}) does not match the number of "
                f"volumes in the data ({self.subj_num * self.sessions * self.volumes})"
            )

    def _load_hmm(self, hmm_file: str):
        try:
            hmm = loadmat(hmm_file)
            self.hmm = self.convert_hmm(hmm)
            self.vpath = np.squeeze(hmm["vpath"])
            self.gamma = np.array(hmm["Gamma"])
            self.Xi = np.array(hmm["Xi"])
        except Exception as e:
            print(f"Error loading HMM file: {e}")

    def _generate_subj_labels_with_session(self):
        subject_session_ids = []
        for subj in self.subj_labels:
            for ses in range(1, self.sessions + 1):
                subject_session_id = f"{subj}_s{ses + 1}"
                subject_session_ids.append(subject_session_id)
        return subject_session_ids

    # get hmm parameters
    @staticmethod
    def convert_hmm(hmm):
        """
        ***Only tested for FMRI DATA***
        Convert the `hmm` object derived from the following command of HMM-MAR Toolbox
        `[hmm, Gamma, Xi, vpath] = hmmmar(f,T,options);`
        to a Python Dictionary for further processing
        """
        hmm_dict = {}
        hmm_param = [
            "train",
            "K",
            "prior",
            "Dir_alpha",
            "Pi",
            "Dir2d_alpha",
            "P",
            "state",
        ]
        train_param = [
            "order",
            "verbose",
            "cyc",
            "initrep",
            "initcyc",
            "standardise",
            "inittype",
            "zeromean",
            "covtype",
            "useParallel",
            "repetitions",
            "Fs",
            "K",
            "onpower",
            "leida",
            "embeddedlags",
            "pca",
            "pca_spatial",
            "lowrank",
            "varimax",
            "maxFOth",
            "pcamar",
            "pcapred",
            "filter",
            "detrend",
            "downsample",
            "leakagecorr",
            "sequential",
            "standardise_pc",
            "regularisation",
            "Gamma_constraint",
            "acrosstrial_constrained",
            "plotGamma",
            "id_mixture",
            "episodic",
            "cluster",
            "initcriterion",
            "meancycstop",
            "cycstogoafterevent",
            "initTestSmallerK",
            "stopcriterion",
            "tol",
            "behaviour",
            "tudamonitoring",
            "tuda",
            "distribution",
            "grouping",
            "Pstructure",
            "Pistructure",
            "dropstates",
            "priorcov_rate",
            "timelag",
            "exptimelag",
            "orderoffset",
            "symmetricprior",
            "uniqueAR",
            "S",
            "prior",
            "Sind",
            "orders",
            "maxorder",
            "DirichletDiag",
            "PriorWeightingP",
            "PriorWeightingPi",
            "hmm",
            "fehist",
            "updateObs",
            "updateGamma",
            "updateP",
            "decodeGamma",
            "keepS_W",
            "useMEX",
            "ndim",
            "active",
        ]
        prior_param = ["Dir2d_alpha", "Dir_alpha"]
        omega_param = ["Gam_shape", "Gam_rate"]
        sub_param = {"train": train_param, "prior": prior_param, "Omega": omega_param}

        hmm = hmm["hmm"][0][0]
        for i, param in enumerate(hmm_param):
            data = hmm[i]

            if param in sub_param.keys():
                data = data[0][0]
                tem_dict = {}

                for j, sub in enumerate(sub_param[param]):
                    tem_dict[sub] = data[j]
                hmm_dict[param] = tem_dict
                if param == "train":
                    if tem_dict["covtype"] in ["sharedfull", "uniquefull"]:
                        hmm_param.append("Omega")
                        prior_param.append("Omega")
                        omega_param.append("Gam_irate")
                        unique = True
                    elif tem_dict["covtype"] in ["shareddiag", "uniquediag"]:
                        hmm_param.append("Omega")
                        prior_param.append("Omega")
                        unique = True
                    else:
                        unique = False

            elif param == "state":
                data = data[0]
                tem_dict = {}
                for k in range(len(data)):
                    pr_dict, w_dict = {}, {}
                    state_data = data[k]
                    prior_data = state_data[0][0][0]
                    state_prior_par = ["sigma", "alpha", "Mean"]
                    w_data = state_data[1][0][0]
                    state_w_par = ["Mu_W", "S_W", "iS_W"]

                    for m, sub_par in enumerate(state_prior_par):
                        pr_dict[sub_par] = prior_data[m]
                    for m, sub_par in enumerate(state_w_par):
                        w_dict[sub_par] = w_data[m]
                    if not unique:
                        omega_data = state_data[2][0][0]
                        state_omega_par = [
                            "Gam_rate",
                            "Gam_shape",
                        ]
                        omega_dict = dict()
                        for m, sub_par in enumerate(state_omega_par):
                            omega_dict[sub_par] = omega_data[m]
                        tem_dict[f"state{k + 1}"] = {
                            "prior": pr_dict,
                            "W": w_dict,
                            "Omega": omega_dict,
                        }
                        continue
                    tem_dict[f"state{k + 1}"] = {
                        "prior": pr_dict,
                        "W": w_dict,
                    }
                hmm_dict[param] = tem_dict
            else:
                hmm_dict[param] = np.squeeze(data)

        return hmm_dict

    # Calculate Vpath features
    def _vpath_fo(self, vpath):
        state_fo = dict()
        length = len(vpath)
        for state in range(1, self.K + 1):
            state_fo[f"state{state}_fo"] = [np.count_nonzero(vpath == state) / length]

        return pd.DataFrame.from_dict(state_fo)

    def _vpath_visit(self, vpath):
        dic = dict()
        length = len(vpath)
        for state in range(1, self.K + 1):
            vpath_bool = vpath == state
            visit = 0
            continuous = False
            for i in range(length):
                if not continuous:
                    if vpath_bool[i]:
                        visit += 1
                    else:
                        continue
                try:
                    if vpath_bool[i + 1]:
                        continuous = True
                    else:
                        continuous = False
                except IndexError:
                    pass
            dic[f"state{state}_visits"] = [visit]

        return pd.DataFrame.from_dict(dic)

    @staticmethod
    def _vpath_switch(vpath):
        dic = dict()
        length = len(vpath)
        change = 0
        for i in range(length - 1):
            if vpath[i] != vpath[i + 1]:
                change += 1
        dic["switch_rate"] = [change / length]

        return pd.DataFrame.from_dict(dic)

    def _vpath_lifetime(self, vpath, mean=True):
        dic = dict()
        length = len(vpath)
        for state in range(1, self.K + 1):
            lifes = []
            vpath_bool = vpath == state
            life = 0
            for i in range(length):
                if vpath_bool[i]:
                    life += 1
                elif (vpath_bool[i] is False) & (life > 0):
                    lifes.append(life)
                    life = 0
                else:
                    pass
            if mean:
                dic[f"state{state}_lifetime"] = [np.mean(lifes)]
            else:
                dic[f"state{state}_lifetime"] = [lifes]

        return pd.DataFrame.from_dict(dic)

    def _vpath_interval(self, vpath, mean=True):
        dic = dict()
        length = len(vpath)
        for state in range(1, self.K + 1):
            intervals = []
            vpath_bool = vpath == state
            interval = 0
            for i in range(length):
                if not vpath_bool[i]:
                    interval += 1
                elif (vpath_bool[i] is True) & (interval > 0):
                    intervals.append(interval)
                    interval = 0
                else:
                    pass
            if mean:
                dic[f"state{state}_interval"] = [np.mean(intervals)]
            else:
                dic[f"state{state}_interval"] = [intervals]

        return pd.DataFrame.from_dict(dic)

    def parse_vpath_chronnectome(self, mean=True):
        """
        Parse chronnectome data and compute various visitation metrics for each subject.

        Parameters
        ----------
        mean : bool, controls if the mean of the interval and lifetime metrics are returned or the raw values

        Returns
        -------
        results : dict
            A dictionary containing DataFrames for each visitation metric, with keys corresponding to the metric names
            The metrics include:
            - vpath_fo: Fractional Occupancy
            - vpath_visit: Visitation counts
            - vpath_lifetime: State lifetimes
            - vpath_interval: State intervals
            - vpath_switch: Switching rate

        """
        funcs = [
            self._vpath_fo,
            self._vpath_visit,
            self._vpath_switch,
            self._vpath_lifetime,
            self._vpath_interval,
        ]
        results = {}
        for func in funcs:
            dfs = []
            for i in range(self.subj_num):
                subj_vpath = self.vpath[self.volumes * i : self.volumes * (i + 1)]
                if func.__name__ in ["_vpath_lifetime", "_vpath_interval"]:
                    subj_df = func(vpath=subj_vpath, mean=mean)
                else:
                    subj_df = func(vpath=subj_vpath)
                try:
                    subj_df.index = self.roi_labels[i]
                except ValueError:
                    raise ValueError(
                        "The number of subject labels does not match the number of subjects"
                    )
                dfs.append(subj_df)

            results[func.__name__] = pd.concat(dfs)

        return results

    def parse_gamma_chronnectome(self, mean=True):
        return NotImplementedError

    def get_means(self):
        means = []
        for i in range(self.K):
            mean = np.squeeze(self.hmm["state"][f"state{i + 1}"]["W"]["Mu_W"])
            means.append(mean)

        return means

    def _conn(self, k):
        # calculate cov and corr matrix
        state = self.hmm["state"][f"state{k}"]["Omega"]
        ndim = state["Gam_shape"].shape[0]
        cov_mat = state["Gam_rate"] / (state["Gam_shape"] - ndim - 1)

        def correlation_from_covariance(covariance):
            v = np.sqrt(np.diag(covariance))
            outer_v = np.outer(v, v)
            correlation = covariance / outer_v
            correlation[covariance == 0] = 0
            return correlation

        corr_mat = correlation_from_covariance(cov_mat)

        return cov_mat, corr_mat

    def get_conns(self):
        # loop through k
        conns = []
        for k in range(1, self.K + 1):
            _, corr_mat = self._conn(k)
            conns.append(corr_mat)

        return conns

    def plot_means(self, roi_labels=None):
        means = self.get_means()
        fig, axes = plt.subplots(self.K, figsize=(20, self.K * 3))
        plt.title("Mean activation of each state")
        if roi_labels is None:
            roi_labels = self.roi_labels
        for i in range(self.K):
            ax = axes[i]
            mean = means[i]
            bar = np.max(np.abs(means[i])) * 0.9
            sns.heatmap(
                pd.DataFrame(mean, columns=[f"state{i + 1}"]).T,
                index=roi_labels,
                cmap="RdBu_r",
                vmin=-bar,
                vmax=bar,
                ax=ax,
                cbar=True,
            )

        return fig

    def plot_conns(self):
        conns = self.get_conns()
        fig, axes = plt.subplots(int(self.K / 2), 2, figsize=(20, self.K * 5))
        plt.title("Connectivity of each state")
        for i in range(self.K):
            ax = axes[i]
            conn = conns[i]
            plotting.plot_matrix(
                conn,
                figure=fig,
                axes=ax,
                colorbar=True,
                vmax=1,
                vmin=-1,
                reorder=False,
                labels=self.roi_labels,
            )
            ax.set_title(f"State {i + 1}")

        return fig

    @staticmethod
    def plot_vpath(self, subj_index: int or list = None):
        if isinstance(subj_index, int):
            subj_index = [subj_index]
        elif isinstance(subj_index, list):
            if not all(isinstance(x, int) for x in subj_index):
                raise ValueError("subj_index must be a list of integers")
        elif subj_index is None:
            subj_index = range(len(self.subj_labels))
        figs = []
        for sub in subj_index:
            vpath = self.vpath[
                self.volumes * sub : self.volumes * (sub + 1)
            ]  # trial type
            durations = np.ones_like(vpath)
            onset = range(vpath.shape[0])
            model_event = pd.DataFrame(
                {"onset": onset, "duration": durations, "trial_type": vpath}
            )
            figs.append(
                plotting.plot_event(
                    model_event,
                    cmap=None,
                    output_file=None,
                    title=f"Subject {self.subj_labels[sub]}",
                )
            )

        return figs
