# Comparing slogans for a gym campaign

slogans_X = [('Get fit, stay strong!', 0.8), ('Get your transformation started today!', 0.5)]
slogans_Y = [('Shape up, live Better!', 0.6), ('Your fitness journey starts here!', 0.4)]

def evaluate_slogans(slogans_X, slogans_Y):
    wins_X, wins_Y = 0, 0
    for (slogan_X, score_X), (slogan_Y, score_Y) in zip(slogans_X, slogans_Y):
        # Assign one point to X if score X is higher, otherwise to Y
        if score_X > score_Y:
            wins_X += 1
        else:
            wins_Y += 1
    success_rate_X = (wins_X / len(slogans_X)) * 100
    success_rate_Y = (wins_Y / len(slogans_Y)) * 100
    return success_rate_X, success_rate_Y

results = evaluate_slogans(slogans_X, slogans_Y)
print(f"The resulting scores are {results}")