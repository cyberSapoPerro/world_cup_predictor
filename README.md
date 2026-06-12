# World Cup Predictor

This is a simple predictor for FIFA World Cup matches that implements a probabilistic Poisson's model based on ELO rating differences to calculate the most likely outcomes.

## Probabilistic Model

The model uses Poisson distributions to calculate the probability each team scores a given number of goals.

Let X be the random variable representing the number of goals scored by team A, and Y the random variable representing the number of goals scored by team B.

We asume that

$$
X \sim \text{Poiss}(\lambda_{1}), \quad Y \sim \text{Poiss}(\lambda_{2})
$$

To estimate the two goal rates, two constraints were used.

The first constraint computes the probability of team A victory according to the ELO rating:

$$
p_{A} = \frac{1}{1 + 10^{-\left( R_{A} - R_{B} \right)/400}}
$$

where $R_{A}, R_{B}$ are the ELO ratings of team A and team B, respectively.

Then constraint is then

$$
\mathbb{P} (X > Y) = p_{A}
$$

The second constraint asumes that the expected number of goals per match in this World Cup is 2.6. So,

$$
\mathbb{E} \left[ X + Y \right] = \lambda_{1} + \lambda_{2} = 2.6
$$

Then, a root finder algorithm is used to get the to parameters of the distributions.

Finally, the most likely match outcomes are computed from the resulting distributions.

## Data

The data is extracted from [this open web site](https://eloratings.net/).

## Usage

Example:

```bash
uv run main.py COL ARG
```
