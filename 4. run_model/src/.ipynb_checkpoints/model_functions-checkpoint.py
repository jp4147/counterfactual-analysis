import configs as c
states = c.states
age_range = c.age_range
import numpy as np
import copy

def get_alive_pop(state):
    '''Returns the proportion of alive patients at the start of each year
       before any transitions are performed'''
    aliveArr = np.zeros((len(age_range)))
    # Everyone starts off as alive in first year
    aliveArr[0] = 1

    age_counter = 1
    for i in range(13,len(age_range) * 12, 12):
        alive_sum = state[i, states['NORM']:states['DT4'] + 1].sum()
        aliveArr[age_counter] += alive_sum
        age_counter += 1
    return aliveArr

def get_incidence(state, t1_cases, t2_cases, t3_cases, t4_cases, cancer_deaths):
    '''Returns incidence of cancer and cancer deaths'''
    # Get alive population at the start of each year
    aliveArr = get_alive_pop(state)

    # Cancer incidence
    cancer_t1 = (np.array(t1_cases) / aliveArr)*100000
    cancer_t2 = (np.array(t2_cases) / aliveArr)*100000
    cancer_t3 = (np.array(t3_cases) / aliveArr)*100000
    cancer_t4 = (np.array(t4_cases) / aliveArr)*100000

    # Cancer death
    cancer_death = (np.array(cancer_deaths) / aliveArr)*100000

    return cancer_t1, cancer_t2, cancer_t3, cancer_t4, cancer_death

def ageATtime(idx):
    return int(np.floor((idx-1)/12)+age_range[0])

def final_alive_age(lst):
    prev_state = 'NORM'
    record = {}
    for i in range(len(lst)):
        if lst[i]!=prev_state:
            record[lst[i-1]] = ageATtime(i-1)
            prev_state = lst[i]
    if lst[i] not in ['ACMORT', 'PCMORT']:
        alive_age = 85
    else:
        alive_age = list(record.values())[-1]
    return alive_age 

def state2array(sh):
    st_arry = np.zeros((len(sh), 11))
    i=0
    for s in sh:
        st_arry[i][states[s]]=1
        i += 1
    return st_arry

def incidence_age(lst, case):
    if case in lst:
        idx = lst.index(case)
        age = int(np.floor((idx-1)/12)+18) # age 18 to 99. ((99-18)+1)*12 = 984. But total number of rows is 985 because the first row is the starting point.
    else:
        age = 0
    return age

def set_start_state():
    '''Defines starting state for the markov model
       Everyone starts off in NORM'''
    start_state = np.zeros(states['TOTAL'])
    start_state[states['NORM']] = 1
    return start_state

def fixed_tp2dict(params_df):
    params_dict = {}
    for age in age_range:
        params_dict[age] = params_df.copy(deep=True)
    return params_dict

def grouping(deaths):
    '''Convert cancer mortality rate by each age into age groups
       15-19, 20-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84
       Input is a numpy array'''
    # For SEER mortality
    lst = []
    # Group 18-19 as 15-19
    lst.append(deaths[0:2].mean())
    # Group 20-24
    lst.append(deaths[2:7].mean())
    # Group 25-29
    lst.append(deaths[7:12].mean())    
    # Group 30-34
    lst.append(deaths[12:17].mean())
    # Group 35-39
    lst.append(deaths[17:22].mean())
    # Group 40-44
    lst.append(deaths[22:27].mean())
    # Group 45-49
    lst.append(deaths[27:32].mean())   
    # Group 50-54
    lst.append(deaths[32:37].mean())
    # Group 55-59
    lst.append(deaths[37:42].mean())
    # Group 60-64
    lst.append(deaths[42:47].mean())
    # Group 65-69
    lst.append(deaths[47:52].mean())
    # Group 70-74
    lst.append(deaths[52:57].mean())
    # Group 75-79
    lst.append(deaths[57:62].mean())
    # Group 80-84
    lst.append(deaths[62:67].mean())

    # Convert list into numpy array
    newArr = np.array(lst)
    return newArr

def withDecreasing_mort(counter, prev_state, prev_dt_history, curr_t_matrix, dF, ut, dt, mort):
    new_dt_history = {}
    dt_pcmort_rev = 0
    
    dt_state = prev_state[c.states[dt]]
    dt_pcmort_trans = curr_t_matrix[c.states[dt], c.states[mort]]
    dt_acmort_trans = curr_t_matrix[c.states[dt], c.states['ACMORT']]
    
    ut_state = prev_state[c.states[ut]]
    ut_dt_trans = curr_t_matrix[c.states[ut], c.states[dt]]
    
    new_pc_cases = 0
    for i in range(counter + 1):
        if i == 0: #new population going into dt and mort
            new_dt_history[i] = ut_state * ut_dt_trans
        else:
            diagnosed_age = np.floor(18 + (counter - 1)/12) # i = survived months
            f = 1
            if i>=12 and i<132: #i==132 is 11 years
                f = dF[diagnosed_age][int(np.floor(i/12))-1]
            else:
                f=0
            new_dt_history[i] = prev_dt_history[i-1] * (1 - dt_pcmort_trans*f - dt_acmort_trans)
            new_pc_cases += prev_dt_history[i-1] * dt_pcmort_trans*f
    
    new_dtcases = new_dt_history[0]
    new_num_dt = sum(new_dt_history.values())
    
    prev_dt_history = copy.deepcopy(new_dt_history)
    
    return new_dtcases, new_num_dt, new_pc_cases, prev_dt_history

# Goodness-of-fit functions
def gof(obs, exp):
    # chi-squared
    # inputs: numpy arrays of observed and expected values
    chi = ((obs-exp)**2)
    chi_sq = sum(chi)
    return chi_sq.sum()

def acceptance_prob(old_gof, new_gof, T):
    if new_gof < old_gof:
        return 1
    else:
        return np.exp((old_gof - new_gof) / T)

def cancer_progression(tm):
    sj_time = []
    for k, tm_age in tm.items():
        s = (1+tm_age[1][2]*(1+tm_age[2][3]*(1/(1-tm_age[3][3])))*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        sj_time.append(s)
    return sj_time

def sojourn_time(tm):
    sj_time = []
    for k, tm_age in tm.items():
        m15 = 1/(1-tm_age[1][1])
        
        m12 = 1/(1-tm_age[1][1])
        m26 = 1/(1-tm_age[2][2])
        m16 = m12+m26
        
        m13 = (1+tm_age[1][2]*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        m37 = 1/(1-tm_age[3][3])
        m17 = m13+m37
        
        m14 = (1+tm_age[1][2]*(1+tm_age[2][3]*(1/(1-tm_age[3][3])))*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        m48 = 1/(1-tm_age[4][4])
        m18 = m14+m48
        s = np.mean([m15, m16, m17, m18])
        sj_time.append(s)
    return sj_time

def sojourn_timeBystage(tm):
    m15, m16, m17, m18 = {}, {}, {}, {}
    for k, tm_age in tm.items():
        m15[k] = 1/(1-tm_age[1][1])
        
        m12 = 1/(1-tm_age[1][1])
        m26 = 1/(1-tm_age[2][2])
        m16[k] = m12+m26
        
        m13 = (1+tm_age[1][2]*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        m37 = 1/(1-tm_age[3][3])
        m17[k] = m13+m37
        
        m14 = (1+tm_age[1][2]*(1+tm_age[2][3]*(1/(1-tm_age[3][3])))*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        m48 = 1/(1-tm_age[4][4])
        m18[k] = m14+m48
    return m15, m16, m17, m18

def sojourn_timeBystage2(tm):
    m12, m23, m34, = {}, {}, {}
    for k, tm_age in tm.items():
        
        m12[k] = 1/(1-tm_age[1][1])
        m23[k] = 1/(1-tm_age[2][2])
        m34[k] = 1/(1-tm_age[3][3])

    return m12, m23, m34

def sojourn_timeBystage3(tm):
    m12, m13, m14, = {}, {}, {}
    for k, tm_age in tm.items():
        
        m12[k] = 1/(1-tm_age[1][1])
        m13[k] = (1+tm_age[1][2]*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))
        m14[k] = (1+tm_age[1][2]*(1+tm_age[2][3]*(1/(1-tm_age[3][3])))*(1/(1-tm_age[2][2])))*(1/(1-tm_age[1][1]))

    return m12, m13, m14

def sojourn_timeBystage4(tm):
    m15, m26, m37, m48 = {}, {}, {}, {}
    for k, tm_age in tm.items():
        m15[k] = 1/(1-tm_age[1][1])
        m26[k] = 1/(1-tm_age[2][2])
        m37[k] = 1/(1-tm_age[3][3])     
        m48[k] = 1/(1-tm_age[4][4])
    return m15, m26, m37, m48