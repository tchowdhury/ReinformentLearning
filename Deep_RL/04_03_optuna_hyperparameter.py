import optuna
import time

def metric(x, y):  
    time.sleep(.05)
    return 2*(1 - x)**2 + (y - x)**2

study = optuna.create_study()

def objective(trial: optuna.Trial):
    # Declare hyperparameters x and y as uniform
    x = trial.suggest_float('x', -5, 5)
    y = trial.suggest_float('y', -5, 5)
    
    value = metric(x, y)
    return value

# Run 50 trials to optimize the objective function
study.optimize(objective, n_trials=50)

# Visualize the trials using a contour plot
fig = optuna.visualization.plot_contour(study)

fig.show()