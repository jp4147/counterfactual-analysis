import common_functions as use
import configs as c
import numpy as np
import copy

def select_new_params(old_param, step, h, l):
    '''Selects new param within range old_param +/- step%
       step: proportion to change param (between 0 and 1), does not depend on temperature
       old_param: old parameter
       Outputs a new parameter'''
    if old_param - old_param * step < 0:
        low = 0
    else:
        low = old_param - old_param * step
        
    if old_param + old_param * step > 1:
        high = 1
    else: 
        high = old_param + old_param * step
    
    if h > 0:
        high = h
    if l> 0:
        low = l
    new_param = np.random.uniform(low, high)

    return new_param

def change_params(old_params_df, step):
    '''Input: Dataframe of probabilities for each transition
       Output: Dataframe of new probabilities for each transition'''
    # Iterate through each row and select new transition probability
    params = old_params_df.copy(deep=True)
    for index, row in params.iterrows():
        params.loc[index, 'Value'] = select_new_params(row['Value'], step, h=-1, l=-1)    
    return params

# def change_params(old_params_df, step):
#     '''Input: Dataframe of probabilities for each transition
#        Output: Dataframe of new probabilities for each transition'''
#     # Iterate through each row and select new transition probability
#     params = old_params_df.copy(deep=True)
#     for index, row in params.iterrows():
#         if index == 'UT4_DT4':
#             params.loc[index, 'Value'] = select_new_params(row['Value'], step, h=-1, l=0.08)
#         elif index == 'UT3_UT4':
#             params.loc[index, 'Value'] = select_new_params(row['Value'], step, h=-1, l=0.01)
#         else: 
#             params.loc[index, 'Value'] = select_new_params(row['Value'], step, h=-1, l=-1)
#     ut1_dt1 = params.loc['UT1_DT1', 'Value'] 
#     ut2_dt2 = params.loc['UT2_DT2', 'Value'] 
#     ut3_dt3 = params.loc['UT3_DT3', 'Value'] 
#     ut4_dt4 = params.loc['UT4_DT4', 'Value']
    
#     ut1_ut2 = params.loc['UT1_UT2', 'Value'] 
#     ut2_ut3 = params.loc['UT2_UT3', 'Value'] 
#     ut3_ut4 = params.loc['UT3_UT4', 'Value']
    
#     params.loc['UT3_DT3', 'Value'] = select_new_params(row['Value'], step, h=ut4_dt4, l=ut3_dt3)
#     params.loc['UT2_DT2', 'Value'] = select_new_params(row['Value'], step, h=ut3_dt3, l=ut2_dt2)
#     params.loc['UT1_DT1', 'Value'] = select_new_params(row['Value'], step, h=ut2_dt2, l=-1)
    
#     params.loc['UT2_UT3', 'Value'] = select_new_params(row['Value'], step, h=ut3_ut4, l=-1)
#     params.loc['UT1_UT2', 'Value'] = select_new_params(row['Value'], step, h=ut3_ut4, l=-1)
#     return params

def set_transition_matrix(p_transition):
    tm={}
    for age in c.age_range:
        tm[age] = np.zeros((len(c.state_names), len(c.state_names)))
        for start in c.alive_states:
            tm[age][c.states[start], c.states['ACMORT']] = use.annual_prob_to_monthly_prob(c.p_acmort.loc[age].item())
        for i, row in p_transition[age].iterrows():
            start = row['Start']
            end = row['End']
            tm[age][c.states[start], c.states[end]] = use.annual_prob_to_monthly_prob(row['Value'])

        tm[age][c.states['DT1'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T1'].values[0])
        tm[age][c.states['DT2'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T2'].values[0])
        tm[age][c.states['DT3'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T3'].values[0])
        tm[age][c.states['DT4'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T4'].values[0])

        tm[age][c.states['ACMORT'], c.states['ACMORT']] = 1
        tm[age][c.states['PCMORT'], c.states['PCMORT']] = 1

    for age in c.age_range:
        for s in c.alive_states:
            p_leave = tm[age][c.states[s], :].sum() # sum all probs leaving state
            tm[age][c.states[s], c.states[s]] = 1 - p_leave
    return tm

def calibrate_transition_matrix(p_transition, age_var=0):
    
    step = 0.01
    if age_var == 0:
        params_dfs = copy.deepcopy(p_transition)
                 
        # provide changes once
        p_transition_new = {}
        new_trans = change_params(p_transition[c.age_range[0]], step)
        for age, df in params_dfs.items():
            p_transition_new[age] = new_trans
            
    else:
        params_dfs = copy.deepcopy(p_transition)
        
        p_transition_new = {}
        for age, df in params_dfs.items():
            p_transition_new[age] = change_params(df, step)
    
    tm={}
    for age in c.age_range:
        tm[age] = np.zeros((len(c.state_names), len(c.state_names)))
        for start in c.alive_states:
            tm[age][c.states[start], c.states['ACMORT']] = use.annual_prob_to_monthly_prob(c.p_acmort.loc[age].item())
        for i, row in p_transition_new[age].iterrows():
            start = row['Start']
            end = row['End']
            tm[age][c.states[start], c.states[end]] = use.annual_prob_to_monthly_prob(row['Value'])

        tm[age][c.states['DT1'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T1'].values[0])
        tm[age][c.states['DT2'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T2'].values[0])
        tm[age][c.states['DT3'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T3'].values[0])
        tm[age][c.states['DT4'], c.states['PCMORT']] = use.annual_prob_to_monthly_prob(c.tstage_mort[c.tstage_mort['age']==age]['T4'].values[0])

        tm[age][c.states['ACMORT'], c.states['ACMORT']] = 1
        tm[age][c.states['PCMORT'], c.states['PCMORT']] = 1

    for age in c.age_range:
        for s in c.alive_states:
            p_leave = tm[age][c.states[s], :].sum() # sum all probs leaving state
            tm[age][c.states[s], c.states[s]] = 1 - p_leave
    return tm, p_transition_new

def cancer_progression(tm):
    cp_time = []
    for k,tm_age in tm.items():
        cp = (1+tm_age[1][2]*(1+tm_age[2][3]*(1/(1-tm_age[3][3])))*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        cp_time.append(cp)
    return cp_time

def sojourn_time(tm):
    sj_time = []
    for k,tm_age in tm.items():
        s = np.mean([1/(1-tm_age[1][1]), 1/(1-tm_age[2][2]), 1/(1-tm_age[3][3]), 1/(1-tm_age[4][4])])
        sj_time.append(s)
    return sj_time

def time2csm(tm):
    m5_10,m6_10,m7_10,m8_10 = [],[],[],[]
    for k,tm_age in tm.items():
        m5_10.append(1/(1-tm_age[5][5]))
        m6_10.append(1/(1-tm_age[6][6]))
        m7_10.append(1/(1-tm_age[7][7]))
        m8_10.append(1/(1-tm_age[8][8]))
    return m5_10,m6_10,m7_10,m8_10