
def convert_hmm(hmm):
    """
    ***Only tested for FMRI DATA***
    Convert the `hmm` object derived from the following command of HMM-MAR Toolbox
    `[hmm, Gamma, Xi, vpath] = hmmmar(f,T,options);`
    to a Python Dictionary for further processing
    """
    hmm_dict    = {} 
    hmm_param   = ['train','K','prior','Dir_alpha','Pi','Dir2d_alpha','P','state']
    train_param = ['order', 'verbose', 'cyc', 'initrep', 
                   'initcyc', 'standardise', 'inittype', 'zeromean',
                   'covtype', 'useParallel', 'repetitions', 'Fs', 'K',
                   'onpower', 'leida', 'embeddedlags', 'pca',
                   'pca_spatial', 'lowrank', 'varimax', 'maxFOth', 
                   'pcamar', 'pcapred', 'filter', 'detrend',
                   'downsample', 'leakagecorr', 'sequential', 'standardise_pc',
                   'regularisation', 'Gamma_constraint', 'acrosstrial_constrained', 
                   'plotGamma', 'id_mixture', 'episodic', 'cluster',
                   'initcriterion', 'meancycstop', 'cycstogoafterevent', 'initTestSmallerK',
                   'stopcriterion', 'tol', 'behaviour', 'tudamonitoring',
                   'tuda', 'distribution', 'grouping', 'Pstructure', 
                   'Pistructure', 'dropstates', 'priorcov_rate', 'timelag',
                   'exptimelag', 'orderoffset', 'symmetricprior', 'uniqueAR',
                   'S', 'prior', 'Sind', 'orders', 'maxorder',
                   'DirichletDiag', 'PriorWeightingP', 'PriorWeightingPi', 'hmm', 
                   'fehist', 'updateObs', 'updateGamma', 'updateP', 'decodeGamma',
                   'keepS_W', 'useMEX', 'ndim', 'active']
    prior_param = ['Dir2d_alpha', 'Dir_alpha']
    omega_param = ['Gam_shape', 'Gam_rate']
    sub_param   = {'train':train_param, 'prior': prior_param, 'Omega': omega_param}

    hmm = hmm['hmm'][0][0]
    for i, param in enumerate(hmm_param):
        data = hmm[i]
  
        if param in sub_param.keys():
            data = data[0][0]
            tem_dict = {}

            for j, sub in enumerate(sub_param[param]):
                tem_dict[sub] = data[j]
            hmm_dict[param] = tem_dict
            if param =='train':
                if tem_dict['covtype'] == 'uniquefull':
                    hmm_param.append('Omega')
                    prior_param.append('Omega')
                    omega_param.append('Gam_irate')
                    unique = True
                elif tem_dict['covtype'] == 'uniquediag':
                    hmm_param.append('Omega'); prior_param.append('Omega'); unique = True
                else:
                    unique = False
           
        elif param == 'state':
            data = data[0]
            tem_dict = {}
            for k in range(len(data)):
                pr_dict, W_dict = {}, {} 
                state_data = data[k]
                prior_data = state_data[0][0][0]; state_prior_par = ['sigma', 'alpha', 'Mean']
                W_data = state_data[1][0][0]; state_W_par = ['Mu_W', 'S_W', 'iS_W']

                for l, sub_par in enumerate(state_prior_par):
                    pr_dict[sub_par] = prior_data[l]
                for l, sub_par in enumerate(state_W_par):
                    W_dict[sub_par] = W_data[l]
                if not unique:
                    Omega_data = state_data[2][0][0]; state_Omega_par = ['Gam_rate','Gam_shape', ]
                    Omega_dict = dict()
                    for l, sub_par in enumerate(state_Omega_par):
                        Omega_dict[sub_par] = Omega_data[l]    
                    tem_dict[f'state{k+1}'] = {'prior':pr_dict, 'W': W_dict, 'Omega': Omega_dict}
                    continue
                tem_dict[f'state{k+1}'] = {'prior':pr_dict, 'W': W_dict,}
            hmm_dict[param] = tem_dict
        else:
            
            hmm_dict[param] = np.squeeze(data)
            
    return hmm_dict
    
    
    
# Calculate Vpath features
def vpath_fo(vpath, K_state):
    state_fo = dict()
    length = len(vpath)
    for state in range(1, K_state+1):
        state_fo[f'state{state}_fo'] = [np.count_nonzero(vpath==state)/length]

    return pd.DataFrame.from_dict(state_fo)

def vpath_visit(vpath, K_state):
    dic = dict() 
    length = len(vpath)
    for state in range(1, K_state+1):
        vpath_bool = vpath == state
        visit = 0; continuous = False
        for i in range(length): 
            if not continuous:
                if vpath_bool[i] == True:
                    visit += 1
                else:
                    continue
            try:
                if vpath_bool[i+1] == True:
                    continuous = True
                else:
                    continuous = False
            except IndexError:
                pass
        dic[f'state{state}_visits'] = [visit]
    
    return pd.DataFrame.from_dict(dic)

def vpath_switch(vpath, K_state):
    dic = dict()
    length = len(vpath)
    change = 0
    for i in range(length-1):
        if vpath[i] !=vpath[i+1]:
            change += 1
    dic['switch_rate'] = [change/length]
    
    return pd.DataFrame.from_dict(dic)

def vpath_lifetime(vpath, K_state):
    dic = dict() 
    length = len(vpath)
    for state in range(1, K_state+1):
        lifes = []
        vpath_bool = vpath == state
        life = 0
        for i in range(length): 
            if vpath_bool[i] == True:
                life += 1
            elif (vpath_bool[i] == False) & (life >0):
                lifes.append(life)
                life = 0
            else:
                pass
        dic[f'state{state}_lifetime'] = [lifes]
    
    return pd.DataFrame.from_dict(dic)

def vpath_interval(vpath, K_state):
    dic = dict() 
    length = len(vpath)
    for state in range(1, K_state+1):
        intervals = []
        vpath_bool = vpath == state
        interval = 0
        for i in range(length): 
            if vpath_bool[i] == False:
                interval += 1
            elif (vpath_bool[i] == True) & (interval >0):
                intervals.append(interval)
                interval = 0
            else:
                pass
        dic[f'state{state}_interval'] = [intervals]
    
    return pd.DataFrame.from_dict(dic)

def get_mean_chronnectome(df):
    """
    df should contains the state lifetime or interval for each subject
    """
    result = pd.DataFrame(np.zeros_like(df), columns=df.columns, index=df.index)
    for subj in range(len(df.index)):
        for state in range(len(df.columns)):
            vector = np.array(df.iloc[subj,state])
            if vector.size:
                mean_value = vector.mean()
            else:
                mean_value = 0
            result.iloc[subj,state] = mean_value
    return result.astype(float)
    
def parse_chronnectome(vpath, subj_num, K_state, timepoints ):
    """
    Parse chronnectome data and compute various visitation metrics for each subject.

    Parameters
    ----------
    vpath : list or np.ndarray
        A list or NumPy Array containing the visitation paths for all subjects.
    subj_num : int
        The number of subjects/runs in the dataset.
    K_state : int
        The number of states in the HMM model.
    timepoints : int
        The number of timepoints for each subject/run.

    Returns
    -------
    results : dict
        A dictionary containing DataFrames for each visitation metric, with keys corresponding to the metric names and values being the DataFrames. The metrics include:
        - vpath_fo: Fractional Occupancy
        - vpath_visit: Visitation counts
        - vpath_lifetime: State lifetimes
        - vpath_interval: State intervals
        - vpath_switch: Switching rate

    Notes
    -----
    This function parses chronnectome data and computes various visitation metrics for each subject. 
    It takes the visitation paths for all subjects, the number of subjects, the number of states in HMM, and the number of timepoints for each subject as input. 
    The function returns a dictionary containing DataFrames for each visitation metric, with keys corresponding to the metric names and values being the DataFrames.

    """
    funcs = [vpath_fo, vpath_visit, vpath_switch, vpath_lifetime, vpath_interval]
    results={}
    for func in funcs:
        dfs = []
        for i in range(subj_num):
            subj_vpath = vpath[timepoints*i:timepoints*(i+1)]
            subj_df = func(subj_vpath, K_state)
            subj_df.index=[subjects_id[i]]
            dfs.append(subj_df)
        if func.__name__ in ['vpath_lifetime','vpath_interval']:
            results[func.__name__] = get_mean_chronnectome(pd.concat(dfs))
        else:
            results[func.__name__] = pd.concat(dfs)

    
    return results
