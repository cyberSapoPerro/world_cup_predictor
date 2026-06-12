import argparse

import pandas as pd
from scipy.stats import poisson
from scipy.optimize import brentq


def top_scores(xg_home, xg_away, max_goals=10, top_n=6):
    """
    Calculate the most probable outcomes
    """
    results = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):

            prob = (
                poisson.pmf(home_goals, xg_home)
                * poisson.pmf(away_goals, xg_away)
            )

            results.append(
                (home_goals, away_goals, prob)
            )

    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_n]


def elo_win_probability(elo_a, elo_b):
    """
    Compute the probability of team A victory
    """
    return 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))


def poisson_win_probability(lambda_a, lambda_b, max_goals=10):
    """
    P(X > Y)
    """
    p = 0.0

    for x in range(max_goals + 1):
        px = poisson.pmf(x, lambda_a)

        for y in range(x):
            py = poisson.pmf(y, lambda_b)
            p += px * py

    return p


def find_lambdas(target_p, total_goals=2.6):
    """
    Find parameters for the Posisson distributions
    """

    def objective(lambda_a):
        lambda_b = total_goals - lambda_a
        return poisson_win_probability(lambda_a, lambda_b) - target_p

    lambda_a_0 = 0.01
    lambda_a = brentq(
        objective,
        lambda_a_0,
        total_goals - lambda_a_0
    )

    lambda_b = total_goals - lambda_a

    return lambda_a, lambda_b


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("fifac_a", type=str)
    parser.add_argument("fifac_b", type=str)

    args = parser.parse_args()

    elo_df = pd.read_csv("data/elo_fifa.csv")
    elo_dict = dict(zip(elo_df["fifa_code"], elo_df["elo"]))

    elo_a = elo_dict[args.fifac_a]
    elo_b = elo_dict[args.fifac_b]

    p_elo = elo_win_probability(elo_a, elo_b)

    lambda_a, lambda_b = find_lambdas(target_p=p_elo,)

    top = top_scores(lambda_a, lambda_b)

    for h, a, p in top:
        print(f"{h}-{a}: {100*p:.2f}%")

if __name__ == "__main__":
    main()
