import pandas as pd
import pickle

############################################################################
age_range = range(18,85)
states = {
    'NORM' : 0,
    'UT1' : 1, 
    'UT2' : 2, 
    'UT3' : 3, 
    'UT4' : 4, 
    'DT1' : 5, 
    'DT2' : 6,
    'DT3' : 7,
    'DT4' : 8,
    'ACMORT' : 9,
    'PCMORT' : 10,
    'TOTAL' : 11
}

age_group = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
       '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84']

state_names = ['NORM', 'UT1', 'UT2', 'UT3', 'UT4', 'DT1', 'DT2', 'DT3', 'DT4','ACMORT', 'PCMORT']
alive_states = ['NORM', 'UT1', 'UT2', 'UT3', 'UT4', 'DT1', 'DT2', 'DT3', 'DT4']
###############################################################################################
# File locations
seer_csm_path = '../../data/targets/cancer_specific_mortality.pickle'
pcmort_target_path = '../../data/targets/SEER_mortality.xlsx'
acmort_target_path = '../../data/ac_mort/cdc_acmort_totalpop_2017.xlsx'
time36 = '../../data/targets/time36_ver23.pickle'
inc_target_path = '../../data/targets/smooth_inc.csv'
# p_transition_path = '../../data/pc_transition_probs.xlsx'
# decreasing_mort_factor_path = 'decreasing factor.pickle'

#Targets
p_acmort = pd.read_excel(acmort_target_path, header=None, index_col=0)
    
with open(seer_csm_path, 'rb') as handle:
    csm = pickle.load(handle)
tstage_mort = pd.DataFrame.from_dict(csm)
tstage_mort['age'] = list(age_range)
    
target_pcmort = pd.read_excel(pcmort_target_path)
target_pcmort=target_pcmort.iloc[:,1:]
target_pcmort.columns=['age','overall_mort']
target_pcmort = target_pcmort['overall_mort'][4:-1]

target_inc = pd.read_csv(inc_target_path)
    
with open(time36, 'rb') as handle:
    target_sojourn = pickle.load(handle)
        
# #Decreasing mortality
# with open(decreasing_mort_factor_path, 'rb') as handle:
#     d_factor = pickle.load(handle)


