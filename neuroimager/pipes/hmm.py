# Date: 2023-06
# Author: Jianzhang Ni (weitqe@GitHub), UoB & CUHK

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting


class HmmParser(object):
    def __init__(
        self,
        hmm: str or dict,
        volumes: int,
        subj_num: int,
        sessions: int = 1,
        vpath=None,
        gamma=None,
        xi=None,
        subj_labels=None,
        roi_labels=None,
        auto_parse=True,
        generate_report=True,
        output_dir=None,
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
        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
        self._check_hmm()
        self.K = int(self.hmm["K"])
        self.states_info = {}
        if auto_parse:
            self.chronnectome = self.parse_chronnectome()
            for i, (mean, conn) in enumerate(zip(self.get_means(), self.get_conns())):
                self.states_info[f"state{i}"] = {"mean": mean, "conn": conn}
            self.P = self.transition_matrix()
        if generate_report:
            self.generate_report()

    """
    *****************************************************************************
    Part 1:  prepare the object
    *****************************************************************************
    """

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
    """
    *****************************************************************************
    Part 2:  prepare the object
    *****************************************************************************
    """

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
                    elif unique:
                        tem_dict[f"state{k + 1}"] = {
                            "prior": pr_dict,
                            "W": w_dict,
                        }
                hmm_dict[param] = tem_dict
            else:
                hmm_dict[param] = np.squeeze(data)

        return hmm_dict

    """
    *****************************************************************************
    Part 3:  Calculate chronnectome features
    *****************************************************************************
    """

    # Calculate Vpath features
    def vpath_fo(self, vpath):
        state_fo = dict()
        length = len(vpath)
        k_states = len(np.unique(vpath))
        max_fo = 0
        max_state = 0
        for state in range(1, k_states + 1):
            fo = [np.count_nonzero(vpath == state) / length]
            if fo == np.nan:
                fo = 0
            state_fo[f"state{state}_fo"] = fo
            if state == 1:
                max_fo = fo
                max_state = state
            else:
                if fo > max_fo:
                    max_fo = fo
                    max_state = state
        state_fo["vpath_max_fo"] = max_fo
        return pd.DataFrame.from_dict(state_fo)

    @staticmethod
    def gamma_fo(gamma):
        gamma_fo_array = gamma.sum(axis=0) / gamma.shape[0]
        state_names = [f"state{i + 1}_fo" for i in range(gamma.shape[1])]
        gamma_fo_df = pd.DataFrame(gamma_fo_array, index=state_names).T
        gamma_fo_df["gamma_max_fo"] = np.max(gamma_fo_array)

        return gamma_fo_df

    def vpath_visit(self, vpath):
        dic = dict()
        length = len(vpath)
        k_states = self.K
        for state in range(1, k_states + 1):
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
    def vpath_switch(vpath):
        dic = dict()
        length = len(vpath)
        change = 0
        for i in range(length - 1):
            if vpath[i] != vpath[i + 1]:
                change += 1
        dic["switch_rate"] = [change / length]

        return pd.DataFrame.from_dict(dic)

    def vpath_lifetime(self, vpath, mean=True):
        dic = dict()
        length = len(vpath)
        k_states = self.K
        for state in range(1, k_states + 1):
            lifes = []
            vpath_bool = vpath == state
            life = 0
            for i in range(length):
                if vpath_bool[i]:
                    life += 1
                elif (vpath_bool[i] == False) & (life > 0):
                    lifes.append(life)
                    life = 0
            if len(lifes) == 0:
                dic[f"state{state}_lifetime"] = [0]
            elif mean:
                dic[f"state{state}_lifetime"] = [np.mean(lifes)]
            else:
                dic[f"state{state}_lifetime"] = [lifes]

        return pd.DataFrame.from_dict(dic)

    def vpath_interval(self, vpath, mean=True):
        dic = dict()
        length = len(vpath)
        k_states = self.K
        for state in range(1, k_states + 1):
            intervals = []
            vpath_bool = vpath == state
            interval = 0
            for i in range(length):
                if not vpath_bool[i]:
                    interval += 1
                elif (vpath_bool[i] == True) & (interval > 0):
                    intervals.append(interval)
                    interval = 0
            if len(intervals) == 0:
                dic[f"state{state}_interval"] = [0]
            elif mean:
                dic[f"state{state}_interval"] = [np.mean(intervals)]
            else:
                dic[f"state{state}_interval"] = [intervals]

        return pd.DataFrame.from_dict(dic)

    def parse_chronnectome(self, mean=True):
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
            self.vpath_fo,
            self.gamma_fo,
            self.vpath_visit,
            self.vpath_switch,
            self.vpath_lifetime,
            self.vpath_interval,
        ]
        results = {}
        for func in funcs:
            dfs = []
            for i in range(self.subj_num):
                subj_vpath = self.vpath[self.volumes * i : self.volumes * (i + 1)]
                subj_gamma = self.gamma[self.volumes * i : self.volumes * (i + 1), :]
                if func.__name__ in ["vpath_lifetime", "vpath_interval"]:
                    subj_df = func(vpath=subj_vpath, mean=mean)
                elif func.__name__ in ["gamma_fo"]:
                    subj_df = func(subj_gamma)
                else:
                    subj_df = func(vpath=subj_vpath)
                try:
                    subj_df.index = [self.subj_labels[i]]
                except ValueError:
                    raise ValueError(
                        "The number of subject labels does not match the number of subjects"
                    )
                dfs.append(subj_df)

            results[func.__name__] = pd.concat(dfs)
        chronnectome_df = pd.concat(
            [
                results["vpath_fo"],
                results["gamma_fo"],
                results["vpath_visit"],
                results["vpath_lifetime"],
                results["vpath_interval"],
                results["vpath_switch"],
            ],
            axis=1,
        )
        return chronnectome_df

    """
    *****************************************************************************
    Part 4:  Calculate state features
    *****************************************************************************
    """

    def get_means(self):
        means = []
        for i in range(self.K):
            mean = np.squeeze(self.hmm["state"][f"state{i + 1}"]["W"]["Mu_W"])
            means.append(mean)

        return means

    def _conn(self, k):
        # calculate cov and corr matrix
        """
        This function only implements the third part of the following matlab function:
        References:
        if do_HMM_pca
            if ~original_space
                warning('Connectivity maps will necessarily be in original space')
            end
            ndim = size(hmm.state(k).W.Mu_W,1);
            covmat = hmm.state(k).W.Mu_W * hmm.state(k).W.Mu_W' + ...
                hmm.Omega.Gam_rate / hmm.Omega.Gam_shape * eye(ndim);
            icovmat = - inv(covmat);
            icovmat = (icovmat ./ repmat(sqrt(abs(diag(icovmat))),1,ndim)) ...
                ./ repmat(sqrt(abs(diag(icovmat)))',ndim,1);
            icovmat(eye(ndim)>0) = 0;
        elseif is_diagonal
            covmat = diag( hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-1) );
            if ~isfield(hmm.state(k).Omega,'Gam_irate')
                hmm.state(k).Omega.Gam_irate = 1 ./ hmm.state(k).Omega.Gam_irate;
            end
            icovmat = diag( hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-1) );
        else
            ndim = length(hmm.state(k).Omega.Gam_rate);
            covmat = hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-ndim-1);
            if ~isfield(hmm.state(k).Omega,'Gam_irate')
                hmm.state(k).Omega.Gam_irate = inv(hmm.state(k).Omega.Gam_rate);
            end
            icovmat = hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-ndim-1);
            icovmat = (icovmat ./ repmat(sqrt(abs(diag(icovmat))),1,ndim)) ...
                ./ repmat(sqrt(abs(diag(icovmat)))',ndim,1);
            icovmat(eye(ndim)>0) = 0;
        end
        """
        if self.hmm["train"]["covtype"] in [
            "sharedfull",
            "uniquefull",
            "shareddiag",
            "uniquediag",
        ]:
            state = self.hmm["Omega"]
        else:
            state = self.hmm["state"][f"state{k}"]["Omega"]

        if self.hmm["train"]["covtype"] in ["diag", "shareddiag", "uniquediag"]:
            cov_mat = np.diag(state["Gam_rate"] / (state["Gam_shape"] - 1))
            if "Gam_irate" not in state.keys():
                state["Gam_irate"] = 1 / state["Gam_rate"]
            icov_mat = np.diag(state["Gam_irate"] * (state["Gam_shape"] - 1))
        else:
            ndim = state["Gam_rate"].shape[0]
            cov_mat = state["Gam_rate"] / (state["Gam_shape"] - ndim - 1)
            if "Gam_irate" not in state.keys():
                state["Gam_irate"] = np.linalg.inv(state["Gam_rate"])
            icov_mat = state["Gam_irate"] * (state["Gam_shape"] - ndim - 1)
            icov_mat = icov_mat / np.sqrt(np.abs(np.diag(icov_mat)))
            icov_mat = icov_mat / np.sqrt(np.abs(np.diag(icov_mat))).T
            np.fill_diagonal(icov_mat, 0)

        def correlation_from_covariance(covariance):
            v = np.sqrt(np.diag(covariance))
            outer_v = np.outer(v, v)
            correlation = covariance / outer_v
            correlation[covariance == 0] = 0
            return correlation

        corr_mat = correlation_from_covariance(cov_mat)

        return cov_mat, icov_mat, corr_mat

    def get_conns(self):
        # loop through k
        conns = []
        for k in range(1, self.K + 1):
            _, _, corr_mat = self._conn(k)
            conns.append(corr_mat)

        return conns

    """
    *****************************************************************************
    Part 5:  Calculate graph features
    *****************************************************************************
    """

    def transition_matrix(self):
        """
        Calculate the transition matrix of the HMM model

        Notes
        This function doesn't consider multiple sessions as the PNAS paper does.
        Since it is just a method to reorder the transition matrix.
        Feel free to use any other clustering method.
        Returns
        -------
        transition_matrix : pd.DataFrame
            The reordered non-stationary transition matrix of the HMM model
        """
        transition_matrix = self.hmm["P"]
        for j in range(self.K):
            transition_matrix[j, j] = 0
            transition_matrix[j, :] = transition_matrix[j, :] / np.sum(
                transition_matrix[j, :]
            )
        state_labels = [f"state{i + 1}" for i in range(self.K)]
        transition_matrix = pd.DataFrame(
            transition_matrix, index=state_labels, columns=state_labels
        )

        # gamma_sub_mean = np.mean(self.gamma, axis=1).squeeze()

        pca = PCA(n_components=1)
        pca1 = pca.fit_transform(self.gamma.T)
        order = np.argsort(pca1, axis=0).ravel()
        transition_matrix = transition_matrix.iloc[order, order]

        return transition_matrix

    def get_graph(self, threshold=0.2):
        import networkx as nx

        if self.P is None:
            self.P = self.transition_matrix()
        graph_data = self.P.values
        graph_data[graph_data < threshold] = 0
        graph_data[graph_data >= threshold] = 1
        graph = nx.DiGraph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        for node in self.P.columns:
            for other_node in self.P.index:
                if node == other_node:
                    continue
                graph.add_weighted_edges_from(
                    [(node, other_node, self.P.loc[node, other_node])]
                )

        return graph

    """
    *****************************************************************************
    Part 6:  Visualization
    *****************************************************************************
    """

    def plot_means(self, roi_labels=None):
        means = self.get_means()
        fig, axes = plt.subplots(self.K, figsize=(len(means) * 2.5, self.K * 1.2))
        fig.suptitle("Mean activation of each state")
        if roi_labels is None:
            roi_labels = self.roi_labels
        for i in range(self.K):
            ax = axes[i]
            mean = means[i]
            bar = np.max(np.abs(means[i])) * 0.9
            sns.heatmap(
                pd.DataFrame(mean, columns=[f"state{i + 1}"], index=roi_labels).T,
                cmap="RdBu_r",
                vmin=-bar,
                vmax=bar,
                ax=ax,
                cbar=True,
            )
        plt.subplots_adjust(hspace=1, wspace=0.5)

        return fig

    def plot_conns(self):
        conns = self.get_conns()
        dim = conns[0].shape[0]
        if self.K % 2 == 1:
            row = int((self.K // 2) + 1)
        elif self.K % 2 == 0:
            row = int(self.K // 2)
        fig, axes = plt.subplots(row, 2, figsize=(12, 6 * int(self.K / 2)))
        axes = axes.ravel()
        fig.suptitle("Connectivity of each state")
        for i in range(self.K):
            ax = axes[i]
            conn = conns[i]
            plotting.plot_matrix(
                conn,
                axes=ax,
                colorbar=True,
                # vmax=1,
                # vmin=-1,
                reorder=False,
            )
            ax.set_title(f"State {i + 1}")
        plt.subplots_adjust(
            hspace=0.5, wspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9
        )

        return fig

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
                )
            )
        if len(figs) == 1:
            return figs[0]
        return figs

    def plot_louvain_community(self, threshold=0.2):
        import networkx as nx
        from sknetwork.clustering import Louvain
        from sknetwork.visualization import svg_graph

        graph = self.get_graph(threshold=threshold)
        sparse_graph = nx.to_scipy_sparse_array(graph)
        # not sure why the sparse array is not working with Louvain, have to use numpy array now.
        numpy_array = sparse_graph.toarray()
        louvain = Louvain()
        labels = louvain.fit_transform(numpy_array)
        labels_array = np.array(labels.todense())
        labels_list = list(np.argmax(labels_array, axis=1))
        names = self.P.columns
        pos = nx.spring_layout(graph)
        fig, ax = plt.subplots(figsize=(self.K, self.K))

        # Set colors for nodes using the labels
        cmap = sns.color_palette("husl", len(labels_list))
        node_colors = [cmap[label] for label in labels_list]

        nx.draw(graph, pos, node_color=node_colors, with_labels=True, ax=ax)
        ax.set_title("Louvain Community Clustering")
        return fig

    """
    *****************************************************************************
    Part 7:  Automation
    *****************************************************************************
    """

    def generate_report(self, threshold=0.2, plot_vpath=True):
        # plot the figures and save them, then write a html file which cites them
        # plot the figures
        out = self.output_dir + f"hmm_{self.K}states_derivatives/"
        if not os.path.exists(out):
            os.makedirs(out)
        states_out = f"{out}states/"
        vpath_out = f"{out}vpath/"
        graph_out = f"{out}graph/"
        if not os.path.exists(states_out):
            os.makedirs(states_out)
        if not os.path.exists(vpath_out):
            os.makedirs(vpath_out)
        if not os.path.exists(graph_out):
            os.makedirs(graph_out)
        means = self.plot_means()
        means.savefig(f"{states_out}means.png")
        plt.close(means)
        conns = self.plot_conns()
        conns.savefig(f"{states_out}conns.png")
        plt.close(conns)
        graph = self.plot_louvain_community(threshold=threshold)
        graph.savefig(f"{graph_out}Louvain.png")
        plt.close(graph)
        if plot_vpath:
            for i, subj_label in enumerate(self.subj_labels):
                subj_vpath = self.plot_vpath(i)
                subj_vpath.savefig(f"{vpath_out}vpath_{subj_label}.png")
                plt.close(subj_vpath)

        with open(f"{out}report.html", "w") as f:
            f.write(
                f"""
                <html>
                <body>
                <h1>Mean activation of each state</h1>
                <img src="{states_out}means.png" alt="means">
                <h1>Connectivity of each state</h1>
                <img src="{states_out}conns.png" alt="conns">
                """
            )
            f.write(
                f"""
                <h1>graph features</h1>
                <img src="{graph_out}Louvain.png" alt="Louvain">
                <h1>State visitation path</h1>

                """
            )
            # loop through subjects and add the images
            try:
                for i, subj_label in enumerate(self.subj_labels):
                    f.write(
                        f"""
                        <h2>Subject {subj_label}</h2>
                        <img src="{vpath_out}vpath_{subj_label}.png" alt="vpath_{subj_label}">
                        """
                    )
            except FileNotFoundError:
                pass
            f.write(
                """
                </body>
                </html>
                """
            )


class HmmModelSelector(object):
    def __init__(
        self,
        models_dir: str,
        krange: list or range,
        rep_num: int,
        volumes: int,
        subj_num: int,
        sessions: int = 1,
        subj_labels: list = None,
        prefix: str = None,
        output_dir: str = None,
    ):
        self.models_dir = models_dir
        self.krange = krange
        self.rep_num = rep_num
        self.volumes = volumes
        self.suj_num = subj_num
        self.sessions = sessions
        self.prefix = prefix
        if subj_labels:
            self.subj_labels = subj_labels
        else:
            self.subj_labels = [f"subj{i}" for i in range(subj_num)]
        if output_dir is None:
            self.output_dir = os.getcwd() + "/"
        else:
            self.output_dir = output_dir

    """
    *****************************************************************************
    Part 1:  Basic Operations
    *****************************************************************************
    """

    def __parse_selected_model(self, hmm_file):
        hmm = HmmParser(
            hmm_file,
            self.volumes,
            self.suj_num,
            self.sessions,
            auto_parse=False,
            generate_report=False,
        )
        return hmm

    @staticmethod
    def __check_file(hmm_file):
        # check if a file exists
        if not os.path.exists(hmm_file):
            raise FileNotFoundError(f"{hmm_file} not found")
        return True

    """
    *****************************************************************************
    Part 2:  Similarity Matrix
    *****************************************************************************
    """

    @staticmethod
    def get_gamma_similarity(gamma1: np.array, gamma2: np.array):
        """
        Mote: This function is recreated from getGammaSimilarity.m of HMM-MAR toolbox.
        Does not support calculation of averaged gamma2, so it has to be a single trail.

        Computes a measure of similarity between two sets of state time courses.
        These can have different number of states, but they must have to be the same length.
        similarity: the sum of joint probabilities under the optimal state alignment
        gamma2_order: optimal state alignment for gamma2 (uses munkres algorithm)
        gamma2_reordered: the second set of state time courses reordered to match gamma1

        """
        # check input
        if gamma1.shape[0] != gamma2.shape[0]:
            raise ValueError(
                "gamma1 and gamma2 must have the same number of time points"
            )
        if len(gamma1.shape) != 2 or len(gamma2.shape) != 2:
            raise ValueError("gamma1 and gamma2 must be 2D arrays")

        T, K = gamma1.shape

        gamma1_0 = gamma1.copy()
        M = np.zeros((K, K))  # cost

        g = gamma2

        K2 = g.shape[1]

        if K < K2:
            gamma1 = np.hstack((gamma1_0, np.zeros((T, K2 - K))))
            K = K2
        elif K > K2:
            g = np.hstack((g, np.zeros((T, K - K2))))

        for k1 in range(K):
            for k2 in range(K):
                M[k1, k2] = (
                    M[k1, k2] + (T - np.sum(np.minimum(gamma1[:, k1], g[:, k2]))) / T
                )
        from munkres import Munkres

        munkres_solver = Munkres()
        gamma2_order = munkres_solver.compute(M.copy())
        cost = sum(M[i][j] for i, j in gamma2_order)

        similarity = K - cost

        gamma2_reordered = gamma2[:, gamma2_order]

        return similarity, gamma2_order, gamma2_reordered

    def calc_simi_matrix(self, state_k):
        import warnings

        gamma = {}
        for rep in range(1, self.rep_num + 1):
            try:
                if self.prefix is None:
                    hmm_file = f"{self.models_dir}k{state_k}_rep{rep}.mat"
                else:
                    hmm_file = f"{self.models_dir}{self.prefix}_k{state_k}_rep{rep}.mat"
                self.__check_file(hmm_file)
                hmm = self.__parse_selected_model(hmm_file)
                gamma[rep] = hmm.gamma
            except FileNotFoundError:
                warnings.warn(
                    f"Rep{rep} for k{state_k} not found, skipping", UserWarning
                )
                continue
        size = len(gamma.keys())
        simi_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                if i == j:
                    simi_matrix[i, j] = 1
                simi_matrix[i, j], _, _ = self.get_gamma_similarity(
                    gamma[i + 1], gamma[j + 1]
                )
                simi_matrix[j, i] = simi_matrix[i, j]

        return simi_matrix

    @staticmethod
    def __mean_simi(simi_matrix, reps=5):
        return (simi_matrix.sum() - reps) / (reps * (reps - 1))

    def get_mean_simi_df(self):
        mean_simi_df = pd.DataFrame(
            [], index=self.krange, columns=["State", "Mean_simi"]
        )
        for K in self.krange:
            if K == 1:
                mean_simi_df.loc[K, "Mean_simi"] = 1
                mean_simi_df.loc[K, "State"] = f"1 States"
            else:
                simi = self.calc_simi_matrix(K)
                mean_simi_df.loc[K, "Mean_simi"] = self.__mean_simi(simi, simi.shape[0])
                mean_simi_df.loc[K, "State"] = f"{K} States"
        return mean_simi_df

    """
    *****************************************************************************
    Part 3:  MaxFO and Switch rate
    *****************************************************************************
    """

    def __get_subj_chronnectome(self, state_k):
        """
        Get the chronnectome of the model
        """
        vpath_max_fo, gamma_max_fo, switch = {}, {}, {}
        for rep in range(1, self.rep_num + 1):
            try:
                if self.prefix is None:
                    hmm_file = f"{self.models_dir}k{state_k}_rep{rep}.mat"
                else:
                    hmm_file = f"{self.models_dir}{self.prefix}_k{state_k}_rep{rep}.mat"
                self.__check_file(hmm_file)
                hmm = self.__parse_selected_model(hmm_file)
                chronnectome = hmm.parse_chronnectome()
                vpath_max_fo[rep] = chronnectome["vpath_max_fo"].values
                gamma_max_fo[rep] = chronnectome["gamma_max_fo"].values
                switch[rep] = chronnectome["switch_rate"].values
            except FileNotFoundError:
                print(f"Rep{rep} for {state_k} not found, skipping")
                continue
        # convert a dict to dataframe
        vpath_max_fo_df = pd.DataFrame.from_dict(vpath_max_fo)
        gamma_max_fo_df = pd.DataFrame.from_dict(gamma_max_fo)
        switch_df = pd.DataFrame.from_dict(switch)

        return vpath_max_fo_df, gamma_max_fo_df, switch_df

    """
    *****************************************************************************
    Part 4:  Visualization
    *****************************************************************************
    """

    @staticmethod
    def plot_simi_matrix(simi_matrix, **kwargs):
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(simi_matrix, cmap="jet", vmax=1, vmin=-1, **kwargs)
        plt.title(f"Similarity Matrix")
        return fig

    def __plot_chronnectome(self, data, feature_name, ax=None):
        """
        Plot the chronnectome of the model
        Parameters
        ----------
        data: pd.DataFrame with shape (n_subjects, N_reps)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(self.rep_num * 2, 5))
        violin_cmap = sns.color_palette("cool", n_colors=self.rep_num, desat=0.8)
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)
        sns.violinplot(data=data, inner=None, palette=violin_cmap, ax=ax)
        sns.stripplot(data=data, jitter=True, size=3, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Repetition")
        ax.set_ylabel(f"{feature_name}")

        return ax

    def plot_chronnectome(self, state_k):
        from neuroimager.plotting.styler import no_edge

        vpath_max_fo, gamma_max_fo, switch_df = self.__get_subj_chronnectome(state_k)
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        self.__plot_chronnectome(vpath_max_fo, "vpath_max_fo", no_edge(axes[0]))
        self.__plot_chronnectome(gamma_max_fo, "gamma_max_fo", no_edge(axes[1]))
        self.__plot_chronnectome(switch_df, "switch", no_edge(axes[2]))
        fig.suptitle(f"Chronnectome for {state_k} states")

        return fig

    def plot_mean_simi(self):
        mean_simi_df = self.get_mean_simi_df()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        from neuroimager.plotting import no_edge

        ax = no_edge(ax)
        fig.suptitle("Mean state similarity")
        sns.barplot(x="State", y="Mean_simi", data=mean_simi_df, color="gray", ax=ax)
        ax.set_xlabel("State")
        ax.set_ylabel("Mean similarity")
        ax.set_ylim(0.4, 1)
        return fig

    """
    *****************************************************************************
    Part 5:  Automation
    *****************************************************************************
    """

    def auto_parse(self):
        out = self.output_dir + f"model_selection/"
        if not os.path.exists(out):
            os.makedirs(out)
        simi_out = f"{out}simi/"
        if not os.path.exists(simi_out):
            os.makedirs(simi_out)
        chrono_out = f"{out}chrono/"
        if not os.path.exists(chrono_out):
            os.makedirs(chrono_out)
        mean_simi_df_fig = self.plot_mean_simi()
        mean_simi_df_fig.savefig(f"{simi_out}mean_simi.png")
        plt.close(mean_simi_df_fig)
        for state_k in self.krange:
            simi_matrix = self.calc_simi_matrix(state_k)
            fig = self.plot_simi_matrix(simi_matrix)
            fig.savefig(f"{simi_out}k{state_k}.png")
            plt.close(fig)
            vpath_max_fo, gamma_max_fo, switch = self.__get_subj_chronnectome(state_k)
            fig = self.plot_chronnectome(state_k)
            fig.savefig(f"{chrono_out}k{state_k}.png")
            plt.close(fig)

        return

    def generate_report(self):
        self.auto_parse()
        out = self.output_dir + f"model_selection/"
        if not os.path.exists(out):
            os.makedirs(out)
        html_file = f"{out}report.html"
        with open(html_file, "w") as f:
            f.write("<html><body>\n")
            f.write("<h1>Model Selection</h1>\n")
            f.write("<h2>Mean Similarity</h2>\n")
            f.write("<img src='simi/mean_simi.png'>\n")
            f.write("<h2>Similarity Matrix</h2>\n")
            for state_k in self.krange:
                f.write(f"<h3>{state_k} States</h3>\n")
                f.write(f"<img src='simi/k{state_k}.png'>\n")
            f.write("<h2>Chronnectome</h2>\n")
            for state_k in self.krange:
                f.write(f"<h3>{state_k} States</h3>\n")
                f.write(f"<img src='chrono/k{state_k}.png'>\n")
            f.write("</body></html>\n")

        return
