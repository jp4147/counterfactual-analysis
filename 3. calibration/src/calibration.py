# Calibrate transition probabiilities to SEER incidence and mortality
import numpy as np
import pandas as pd
import configs as c
import common_functions as use
import copy
import pickle
from scipy.stats import kde
import transition_matrix as tm
import model_functions as mf

from model import markov_withDM, markov, microsim, counterfactual

# Set simulated annealing parameters
sim_anneal_params = {
    'starting_T': 1.0,
    'final_T': 0.01, # 0.01
    'cooling_rate': 0.9, # 0.9
    'iterations': 100} # 100

target_t1 = np.array(c.target_inc['t1'])
target_t2 = np.array(c.target_inc['t2'])
target_t3 = np.array(c.target_inc['t3'])
target_t4 = np.array(c.target_inc['t4'])

target_pcmort = np.array(c.target_pcmort)
target_sojourn = c.target_sojourn

class sim_anneal:
    def __init__(self, tp_path, age_var, model, save):
        self.age_var = age_var
        #Transition probabilities
        if tp_path[-1] == 'x':
            p_transition = pd.read_excel(tp_path, index_col = 0)
            p_transition = mf.fixed_tp2dict(p_transition)
        else: 
            with open(tp_path, 'rb') as handle:
                p_transition = pickle.load(handle)
        self.p_transition = p_transition
        self.save = save
        self.model = model

    def anneal(self):

        t_matrix = tm.set_transition_matrix(self.p_transition)
        if self.model == 'mk_dm':
            model = markov_withDM(t_matrix, c.d_factor)
        elif self.model == 'mk':
            model = markov(t_matrix)
            
        states_total, cases = model.run_markov()

        dt1_init, dt2_init, dt3_init, dt4_init, pcmort_init = mf.get_incidence(np.array(states_total), cases['dt1'], cases['dt2'], cases['dt3'], cases['dt4'], cases['pcmort'])

        sojourn_init = mf.sojourn_time(t_matrix)

        # Collapse into age groups for mortality
        grouped_pcmort_init = mf.grouping(pcmort_init)

        t1_gof = mf.gof(dt1_init, target_t1)
        t2_gof = mf.gof(dt2_init, target_t2)
        t3_gof = mf.gof(dt3_init, target_t3)
        t4_gof = mf.gof(dt4_init, target_t4)

        pcmort_gof = mf.gof(grouped_pcmort_init, target_pcmort)
        sojourn_gof = mf.gof(sojourn_init, target_sojourn)
        
        # print(pcmort_gof)
        # print(cancer_progression_gof)
        # print(sojourn_gof)
        # print(t1_gof)
        # print(t2_gof)
        # print(t3_gof)
        # print(t4_gof)
        # old_gof = t1_gof + t2_gof + t3_gof + t4_gof + pcmort_gof
        # old_gof = t1_gof + t2_gof + t3_gof + t4_gof + pcmort_gof + sojourn_gof
        old_gof = t1_gof + t2_gof + t3_gof + t4_gof + pcmort_gof

        # Starting temperature
        T = sim_anneal_params['starting_T']
        Tupdate = 2
        # Start temperature loop
        # Annealing schedule
        params_dict = copy.deepcopy(self.p_transition)
        while T > sim_anneal_params['final_T']:

            for i in range(sim_anneal_params['iterations']):

                # Find new values for transition probabilities
                new_params_matrix, new_params_dict = tm.calibrate_transition_matrix(params_dict, age_var=self.age_var)
                # Get new solutions
                if self.model == 'mk_dm':
                    model = markov_withDM(new_params_matrix, c.d_factor)
                elif self.model == 'mk':
                    model = markov(new_params_matrix)

                states_total, cases = model.run_markov()
                dt1_new, dt2_new, dt3_new, dt4_new, pcmort_new = mf.get_incidence(np.array(states_total), cases['dt1'], cases['dt2'], cases['dt3'], cases['dt4'], cases['pcmort'])

                # Progression time and Sojourn time
                sojourn_new = mf.sojourn_time(new_params_matrix)
                grouped_pcmort_new = mf.grouping(pcmort_new)

                # Calculate new gof
                new_t1_gof = mf.gof(dt1_new, target_t1)
                new_t2_gof = mf.gof(dt2_new, target_t2)
                new_t3_gof = mf.gof(dt3_new, target_t3)
                new_t4_gof = mf.gof(dt4_new, target_t4)

                new_pcmort_gof = mf.gof(grouped_pcmort_new, target_pcmort)
                new_sojourn_gof = mf.gof(sojourn_new, target_sojourn)

                # new_gof = new_t1_gof + new_t2_gof + new_t3_gof + new_t4_gof + new_pcmort_gof
                # new_gof = new_t1_gof + new_t2_gof + new_t3_gof + new_t4_gof + new_pcmort_gof + new_sojourn_gof
                new_gof = new_t1_gof +  new_t2_gof + new_t3_gof + new_t4_gof + new_pcmort_gof

                ap = mf.acceptance_prob(old_gof, new_gof, T)

                # # Decide if the new solution is accepted
                if np.random.uniform() < ap:
                    params_dict = new_params_dict
                    old_gof = new_gof
                # if i%100 ==0:
                    print(T, i, new_gof)

            T = T * sim_anneal_params['cooling_rate']
            if len(self.save)>0:
                with open(self.save, 'wb') as handle:
                    pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return params_dict