import argparse

import pandas as pd

from scipy.stats import poisson
from scipy.optimize import brentq

from rich.console import Console
from rich.table import Table


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

    parser.add_argument(
        "--lambdas",
        action="store_true",
        dest="lambdas"
    )

    parser.add_argument("team_a", type=str)
    parser.add_argument("team_b", type=str)

    args = parser.parse_args()

    if args.lambdas == True:
        lambda_a = float(args.team_a)
        lambda_b = float(args.team_b)
        team_a_name = "Team A"
        team_b_name = "Team B"
    else:
        elo_df = pd.read_csv("data/elo_fifa.csv")
        elo_dict = dict(zip(elo_df["fifa_code"], elo_df["elo"]))
        elo_a = elo_dict[args.team_a]
        elo_b = elo_dict[args.team_b]
        p_elo = elo_win_probability(elo_a, elo_b)
        lambda_a, lambda_b = find_lambdas(target_p=p_elo,)

        team_a_name = args.team_a
        team_b_name = args.team_b

    table = Table(title="Predictions")
    table.add_column(team_a_name)
    table.add_column(team_b_name)
    table.add_column("Relative Freq")
    table.add_column("Cumulative Freq")

    top = top_scores(lambda_a, lambda_b)

    acc = 0
    for h, a, p in top:
        acc += p
        table.add_row(str(h), str(a), f"{100*p:.2f}%", f"{100*acc:.2f}%")

    console = Console()
    console.print(table)

if __name__ == "__main__":
    main()
