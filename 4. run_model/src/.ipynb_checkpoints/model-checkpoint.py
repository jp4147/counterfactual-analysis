import configs as c
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import model_functions as mf

states = c.states
age_range = c.age_range
state_names = c.state_names
ageATtime = mf.ageATtime
incidence_age = mf.incidence_age
state2array = mf.state2array

class markov_withDM:
    def __init__(self, tm, d_factor):
        self.tm = tm
        self.d_factor = d_factor
        
    def run_markov(self):
        start_state = mf.set_start_state()
        state={}
        cycle_counter = 0
        t1_cases, t2_cases, t3_cases, t4_cases = {},{},{},{}
        t1_deaths, t2_deaths, t3_deaths, t4_deaths = {},{},{},{}
        cancer_deaths = {}
        for age in age_range:
            curr_t_matrix = self.tm[age]
            t1_cases[age] = 0
            t2_cases[age] = 0
            t3_cases[age] = 0
            t4_cases[age] = 0

            t1_deaths[age] = 0
            t2_deaths[age] = 0
            t3_deaths[age] = 0
            t4_deaths[age] = 0
            
            cancer_deaths[age] = 0
            for month in range(12):
                if age == 18 and month == 0:
                    prev_state = start_state
                    state[cycle_counter] = prev_state
                    prev_num_dt1, prev_num_dt2, prev_num_dt3, prev_num_dt4 = 0, 0, 0, 0
                    prev_dt1_history, prev_dt2_history, prev_dt3_history, prev_dt4_history = {}, {}, {}, {}
                else:
                    prev_state = state[cycle_counter]

                new_state = np.dot(prev_state, curr_t_matrix)

                ut1_dt1, dt1num, dt1_pc, dt1_history = mf.withDecreasing_mort(cycle_counter, prev_state, prev_dt1_history, curr_t_matrix, self.d_factor, 'UT1', 'DT1', 'PCMORT')
                ut2_dt2, dt2num, dt2_pc, dt2_history = mf.withDecreasing_mort(cycle_counter, prev_state, prev_dt2_history, curr_t_matrix, self.d_factor, 'UT2', 'DT2', 'PCMORT')
                ut3_dt3, dt3num, dt3_pc, dt3_history = mf.withDecreasing_mort(cycle_counter, prev_state, prev_dt3_history, curr_t_matrix, self.d_factor, 'UT3', 'DT3', 'PCMORT')
                ut4_dt4, dt4num, dt4_pc, dt4_history = mf.withDecreasing_mort(cycle_counter, prev_state, prev_dt4_history, curr_t_matrix, self.d_factor, 'UT4', 'DT4', 'PCMORT')

                t1_cases[age] += ut1_dt1
                t2_cases[age] += ut2_dt2
                t3_cases[age] += ut3_dt3
                t4_cases[age] += ut4_dt4
                
                cancer_deaths[age] += dt1_pc + dt2_pc + dt3_pc + dt4_pc
                
                prev_dt1_history = copy.deepcopy(dt1_history)
                prev_dt2_history = copy.deepcopy(dt2_history)
                prev_dt3_history = copy.deepcopy(dt3_history)
                prev_dt4_history = copy.deepcopy(dt4_history)

                new_state_rev = copy.deepcopy(new_state)
                new_state_rev[c.states['DT1']] = dt1num
                new_state_rev[c.states['DT2']] = dt2num
                new_state_rev[c.states['DT3']] = dt3num
                new_state_rev[c.states['DT4']] = dt4num
                new_pc_state = prev_state[c.states['PCMORT']] + dt1_pc + dt2_pc + dt3_pc +dt4_pc
                new_state_rev[c.states['PCMORT']] = new_pc_state

                cycle_counter += 1
                state[cycle_counter] = list(new_state_rev)
                # print(state)
                
        # print(state)
        states_total_df = pd.DataFrame.from_dict(state, orient = 'index', columns=[state_names])
        cases = pd.DataFrame.from_dict(t1_cases,orient='index')
        cases.columns = ['dt1']
        cases['dt2'] = pd.DataFrame.from_dict(t2_cases,orient='index')
        cases['dt3'] = pd.DataFrame.from_dict(t3_cases,orient='index')
        cases['dt4'] = pd.DataFrame.from_dict(t4_cases,orient='index')
        cases['pcmort'] = pd.DataFrame.from_dict(cancer_deaths,orient='index')
        # print(state)

        return states_total_df, cases
    
class markov:
    def __init__(self, tm):
        self.tm = tm
        
    def run_markov(self):
        start_state = mf.set_start_state()
        state_collect = np.zeros(((age_range[-1]-age_range[0]+1)*12+1, states['TOTAL']))
        cycle_counter = 0
        t1_cases, t2_cases, t3_cases, t4_cases = {},{},{},{}
        t1_deaths, t2_deaths, t3_deaths, t4_deaths = {},{},{},{}
        cancer_deaths = {}
        for age in age_range:
            curr_tm = self.tm[age]
            t1_cases[age] = 0
            t2_cases[age] = 0
            t3_cases[age] = 0
            t4_cases[age] = 0

            t1_deaths[age] = 0
            t2_deaths[age] = 0
            t3_deaths[age] = 0
            t4_deaths[age] = 0

            cancer_deaths[age] = 0
            for month in range(12):
                if age == 18 and month == 0:
                    prev_state = start_state
                    # print(prev_state)
                    # print(cycle_counter)
                    state_collect[cycle_counter] = prev_state
                else:
                    prev_state = state_collect[cycle_counter]
                new_state = np.dot(prev_state, curr_tm)

                t1_cases[age] += self.tm[age][states['UT1'], states['DT1']] * prev_state[states['UT1']]
                t2_cases[age] += self.tm[age][states['UT2'], states['DT2']] * prev_state[states['UT2']]
                t3_cases[age] += self.tm[age][states['UT3'], states['DT3']] * prev_state[states['UT3']]
                t4_cases[age] += self.tm[age][states['UT4'], states['DT4']] * prev_state[states['UT4']]

                t1_deaths[age] += self.tm[age][states['DT1'], states['PCMORT']]* prev_state[states['DT1']]
                t2_deaths[age] += self.tm[age][states['DT2'], states['PCMORT']] * prev_state[states['DT2']]
                t3_deaths[age] += self.tm[age][states['DT3'], states['PCMORT']] * prev_state[states['DT3']]
                t4_deaths[age] += self.tm[age][states['DT4'], states['PCMORT']] * prev_state[states['DT4']]

                cancer_deaths[age] += self.tm[age][states['DT1'], states['PCMORT']]* prev_state[states['DT1']]
                cancer_deaths[age] += self.tm[age][states['DT2'], states['PCMORT']] * prev_state[states['DT2']]
                cancer_deaths[age] += self.tm[age][states['DT3'], states['PCMORT']] * prev_state[states['DT3']]
                cancer_deaths[age] += self.tm[age][states['DT4'], states['PCMORT']] * prev_state[states['DT4']]

                cycle_counter +=1
                state_collect[cycle_counter] = new_state
                
        states_total_df = pd.DataFrame(state_collect, columns=[state_names])
        cases = pd.DataFrame.from_dict(t1_cases,orient='index')
        cases.columns = ['dt1']
        cases['dt2'] = pd.DataFrame.from_dict(t2_cases,orient='index')
        cases['dt3'] = pd.DataFrame.from_dict(t3_cases,orient='index')
        cases['dt4'] = pd.DataFrame.from_dict(t4_cases,orient='index')
        cases['pcmort'] = pd.DataFrame.from_dict(cancer_deaths,orient='index')

        return states_total_df, cases 
                
                                
class microsim:
    def __init__(self, tm, num):
        self.tm = tm
        self.population_size = num 
        
    def simulate_person(self):
        """Simulates a person's life events and returns the health state at every month of age."""
        # everyone starts from NORM
        curr_state = 'NORM'
        state_history = ['NORM']
        for age in age_range:
            for month in range(12):
                curr_state = np.random.choice(state_names,replace=True,p=self.tm[age][states[curr_state]])
                state_history.append(curr_state)

        return state_history
    
    def simulate_population(self):
        """Simulates a population of people and returns the number of people dead by pancreatic cancer"""
        dt1_cases, dt2_cases, dt3_cases, dt4_cases = {}, {}, {}, {}
        pcmort = {}
        ind_state = []
        for age in age_range:
            dt1_cases[age] = 0
            dt2_cases[age] = 0
            dt3_cases[age] = 0
            dt4_cases[age] = 0
            pcmort[age] = 0

        tot_rows = (age_range[-1] - age_range[0] + 1) * 12 + 1
        states_total = np.zeros((tot_rows, len(state_names)))
        for i in tqdm(range(self.population_size)):
            state_history = self.simulate_person()
            dt1_age = incidence_age(state_history,'DT1')
            dt2_age = incidence_age(state_history,'DT2')
            dt3_age = incidence_age(state_history,'DT3')
            dt4_age = incidence_age(state_history,'DT4')

            pcmort_age = incidence_age(state_history,'PCMORT')

            if dt1_age != 0:
                dt1_cases[dt1_age] += 1
            if dt2_age != 0:
                dt2_cases[dt2_age] += 1
            if dt3_age != 0:
                dt3_cases[dt3_age] += 1
            if dt4_age != 0:
                dt4_cases[dt4_age] += 1
            if pcmort_age != 0:
                pcmort[pcmort_age] += 1

            states_total += state2array(state_history)
            ind_state.append(state_history)
        states_total_df = pd.DataFrame(states_total, columns=[state_names])        
        cases = pd.DataFrame.from_dict(dt1_cases,orient='index')
        cases.columns = ['dt1']
        cases['dt2'] = pd.DataFrame.from_dict(dt2_cases,orient='index')
        cases['dt3'] = pd.DataFrame.from_dict(dt3_cases,orient='index')
        cases['dt4'] = pd.DataFrame.from_dict(dt4_cases,orient='index')
        cases['pcmort'] = pd.DataFrame.from_dict(pcmort,orient='index')
        return states_total_df, cases, ind_state
    
    
class counterfactual:
    def __init__(self, ind_state, tm):
        self.ind_state = ind_state
        self.tm = tm
        
    def cohort(self, included_state, diedBy):
        idx = []
        for i in tqdm(range(len(self.ind_state))):
            for s in included_state:
                if s in self.ind_state[i]:
                    if 'PCMORT' in self.ind_state[i]:
                        idx.append(i)

        cohort4counterfactual = []
        rest = []
        for i in tqdm(range(len(self.ind_state))):
            if i in idx:
                cohort4counterfactual.append(self.ind_state[i])
            else:
                rest.append(self.ind_state[i])
        return [rest, cohort4counterfactual]
    
    def state_summary(self, sh):
        prev_state = 'NORM'
        record = {}
        for i in range(len(sh)):
            if sh[i] != prev_state:
                record[sh[i-1]] = ageATtime(i-1)
                prev_state = sh[i]
        record[sh[i]] = ageATtime(i)
        return record

# counterfactual method 1 (overdiagnosis involved)
    def run_counterfactual(self, cohort):
        no_cf = cohort[0]
        cf_before = cohort[1]
        cf_after = []
        for test in tqdm(cf_before):
            tm_copy1 = copy.deepcopy(self.tm)
            tm_copy2 = copy.deepcopy(self.tm)
            detected_age = ageATtime([test.index(s) for s in test if s.startswith('D')][0])
            idx = len(test) - 1 - test[::-1].index('UT1') #last index when it was UT1
            state_history = test[0:idx]
            intervention_age = ageATtime(idx)
            intervention_month = idx%12
            state_history.append('DT1')
            if intervention_month == 0:
                intervention_age += 1

            curr_state = state_history[-1]
            for age in range(intervention_age, age_range[-1]+1):
                for month in range(12-intervention_month):
                    if age < detected_age:
                        tm_copy1[age][states['DT1'],states['DT1']] = tm_copy1[age][states['DT1'],states['DT1']] + tm_copy1[age][states['DT1'],states['PCMORT']] + tm_copy1[age][states['DT1'],states['ACMORT']]
                        tm_copy1[age][states['DT1'],states['PCMORT']] = 0
                        tm_copy1[age][states['DT1'],states['ACMORT']] = 0
                        curr_state = np.random.choice(state_names,replace=True,p=tm_copy1[age][states[curr_state]])
                    else:
                        curr_state = np.random.choice(state_names,replace=True,p=tm_copy2[age][states[curr_state]])

                    state_history.append(curr_state)
                intervention_month = 0
            cf_after.append(state_history)
            
        new_sh = no_cf + cf_after
        
        dt1_cases, dt2_cases, dt3_cases, dt4_cases = {}, {}, {}, {}
        pcmort = {}
        for i in age_range:
            dt1_cases[i] = 0
            dt2_cases[i] = 0
            dt3_cases[i] = 0
            dt4_cases[i] = 0
            pcmort[i] = 0

        tot_rows = (age_range[-1] - age_range[0] + 1) * 12 + 1
        states_total = np.zeros((tot_rows, len(state_names)))        
        for state_history in tqdm(new_sh):
            dt1_age = incidence_age(state_history,'DT1')
            dt2_age = incidence_age(state_history,'DT2')
            dt3_age = incidence_age(state_history,'DT3')
            dt4_age = incidence_age(state_history,'DT4')

            pcmort_age = incidence_age(state_history,'PCMORT')

            if dt1_age != 0:
                dt1_cases[dt1_age] += 1
            if dt2_age != 0:
                dt2_cases[dt2_age] += 1
            if dt3_age != 0:
                dt3_cases[dt3_age] += 1
            if dt4_age != 0:
                dt4_cases[dt4_age] += 1
            if pcmort_age != 0:
                pcmort[pcmort_age] += 1

            states_total += state2array(state_history)
        states_total_df = pd.DataFrame(states_total, columns=[state_names])
        cases = pd.DataFrame.from_dict(dt1_cases,orient='index')
        cases.columns = ['dt1']
        cases['dt2'] = pd.DataFrame.from_dict(dt2_cases,orient='index')
        cases['dt3'] = pd.DataFrame.from_dict(dt3_cases,orient='index')
        cases['dt4'] = pd.DataFrame.from_dict(dt4_cases,orient='index')
        cases['pcmort'] = pd.DataFrame.from_dict(pcmort,orient='index')
        
        return cf_after, states_total_df, cases

# counterfactual method 2 (result is same as method 1, overdiagnosis involved)
    def run_counterfactual2(self, cohort):
        no_cf = cohort[0]
        cf_before = cohort[1]
        cf_after = []
        for test in tqdm(cf_before):
            tm_copy1 = copy.deepcopy(self.tm)
            tm_copy2 = copy.deepcopy(self.tm)
            detected_month = [test.index(s) for s in test if s.startswith('D')][0]
            idx = len(test) - 1 - test[::-1].index('UT1') #last index when it was UT1
            state_history = test[0:idx]
            
            stage = state_history[-1][-1]
            if stage == 'M':
                stage ='1'
            state_history.append('DT'+ stage)
            curr_state = state_history[-1]
            for t in range(len(state_history), len(c.age_range)*12+1):
                age = ageATtime(t)
                if t < detected_month:
                    tm_copy1[age][states['DT'+stage],states['DT'+stage]] = tm_copy1[age][states['DT'+stage],states['DT'+stage]] + tm_copy1[age][states['DT'+stage],states['PCMORT']] + tm_copy1[age][states['DT'+stage],states['ACMORT']]
                    tm_copy1[age][states['DT'+stage],states['PCMORT']] = 0
                    tm_copy1[age][states['DT'+stage],states['ACMORT']] = 0
                    curr_state = np.random.choice(state_names,replace=True,p=tm_copy1[age][states[curr_state]])
                else:
                    curr_state = np.random.choice(state_names,replace=True,p=tm_copy2[age][states[curr_state]])

                state_history.append(curr_state)

            cf_after.append(state_history)
            
        new_sh = no_cf + cf_after
        
        dt1_cases, dt2_cases, dt3_cases, dt4_cases = {}, {}, {}, {}
        pcmort = {}
        for i in age_range:
            dt1_cases[i] = 0
            dt2_cases[i] = 0
            dt3_cases[i] = 0
            dt4_cases[i] = 0
            pcmort[i] = 0

        tot_rows = (age_range[-1] - age_range[0] + 1) * 12 + 1
        states_total = np.zeros((tot_rows, len(state_names)))        
        for state_history in tqdm(new_sh):
            dt1_age = incidence_age(state_history,'DT1')
            dt2_age = incidence_age(state_history,'DT2')
            dt3_age = incidence_age(state_history,'DT3')
            dt4_age = incidence_age(state_history,'DT4')

            pcmort_age = incidence_age(state_history,'PCMORT')

            if dt1_age != 0:
                dt1_cases[dt1_age] += 1
            if dt2_age != 0:
                dt2_cases[dt2_age] += 1
            if dt3_age != 0:
                dt3_cases[dt3_age] += 1
            if dt4_age != 0:
                dt4_cases[dt4_age] += 1
            if pcmort_age != 0:
                pcmort[pcmort_age] += 1

            states_total += state2array(state_history)
        states_total_df = pd.DataFrame(states_total, columns=[state_names])
        cases = pd.DataFrame.from_dict(dt1_cases,orient='index')
        cases.columns = ['dt1']
        cases['dt2'] = pd.DataFrame.from_dict(dt2_cases,orient='index')
        cases['dt3'] = pd.DataFrame.from_dict(dt3_cases,orient='index')
        cases['dt4'] = pd.DataFrame.from_dict(dt4_cases,orient='index')
        cases['pcmort'] = pd.DataFrame.from_dict(pcmort,orient='index')
        
        return cf_after, states_total_df, cases

    def run_counterfactual_interventionBymonths(self, cohort, months_earlier, died_by):
        no_cf = cohort[0]
        selected_cohort = cohort[1]
        not_eligible, cf_before = [],[]
        for test in selected_cohort:
            detected_month = [test.index(s) for s in test if s.startswith('D')][0]
            early_detection = detected_month-months_earlier
            state_history = test[0:early_detection+1]
            if early_detection>0 and state_history[-1] != 'NORM':
                cf_before.append(test)
            else:
                not_eligible.append(test)

        cf_after = []
        for test in cf_before:

            # test = cf_before[0]
            tm_copy1 = copy.deepcopy(self.tm)
            tm_copy2 = copy.deepcopy(self.tm)
            detected_month = [test.index(s) for s in test if s.startswith('D')][0]
            died = [test.index(s) for s in test if s.startswith(died_by[0])]
            stay_alive = died[0]
                
            early_detection = detected_month-months_earlier
            state_history = test[0:early_detection+1]

            stage = state_history[-1][-1]
            state_history.append('DT'+ stage)
            curr_state = state_history[-1]
            for t in range(len(state_history), len(c.age_range)*12+1):
                age = ageATtime(t)
                if t< stay_alive:
                # if t < detected_month:
                    tm_copy1[age][states['DT'+stage],states['DT'+stage]] = tm_copy1[age][states['DT'+stage],states['DT'+stage]] + tm_copy1[age][states['DT'+stage],states['PCMORT']] + tm_copy1[age][states['DT'+stage],states['ACMORT']]
                    tm_copy1[age][states['DT'+stage],states['PCMORT']] = 0
                    tm_copy1[age][states['DT'+stage],states['ACMORT']] = 0
                    curr_state = np.random.choice(state_names,replace=True,p=tm_copy1[age][states[curr_state]])
                else:
                    curr_state = np.random.choice(state_names,replace=True,p=tm_copy2[age][states[curr_state]])

                state_history.append(curr_state)

            cf_after.append(state_history)
        return cf_before, cf_after