import numpy as np

"""
Calculating discounted returns for agent strategies
Discounted returns help in evaluating the total amount of rewards an agent can expect to accumulate over time, taking into account 
that future rewards are less valuable than immediate rewards. You are given the expected rewards for two different 
strategies (exp_rewards_strategy_1 and exp_rewards_strategy_2) of an RL agent. Your task is to calculate the discounted 
return for each strategy and determine which one yields the higher return.
"""

exp_rewards_strategy_1 = np.array([3, 2, -1, 5])

discount_factor = 0.9

# Compute discounts
discounts_strategy_1 = np.array([discount_factor ** i for i in range(len(exp_rewards_strategy_1))])

# Compute the discounted return
discounted_return_strategy_1 = np.sum(exp_rewards_strategy_1 * discounts_strategy_1)

print(f"The discounted return of the first strategy is {discounted_return_strategy_1}") 

exp_rewards_strategy_2 = np.array([6, -5, -3, -2])

discount_factor = 0.9

# Compute discounts
discounts_strategy_2 = np.array([discount_factor ** i for i in range(len(exp_rewards_strategy_2))]) 

# Compute the discounted return
discounted_return_strategy_2 = np.sum(exp_rewards_strategy_2 * discounts_strategy_2)

print(f"The discounted return of the second strategy is {discounted_return_strategy_2}")