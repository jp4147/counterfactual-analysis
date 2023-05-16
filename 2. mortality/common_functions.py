import numpy as np
import pandas as pd

def rate_to_prob(rate, time):
	# Converts rate into a probability
    prob = 1 - np.exp(-abs(rate) * time)
    return prob

def prob_to_rate(prob, time):
	# Converts probability to rate
    rate = -(np.log(1 - prob)) / time
    return rate

def annual_prob_to_monthly_prob(yearly_prob):
    # Converts annual probability to monthly probability
    return 1 - (1 - yearly_prob) ** (1/12)
