import numpy as np
import pandas as pd


def pip_alg(u: int,
            l: int,
            delta: float,
            b: int) -> int:
    '''
    Sun et. al.
    returns number of days to rent for
    '''
    if u <= b:
        return b
    elif b <= l:
        return b * min(1, np.sqrt(delta / (1 - delta)))
    else:
        if delta > l / (l + b):
            zeta = 1 + b / l
        else:
            zeta = delta + b * (1 - delta) / l + 2 * np.sqrt(delta * b * (1 - delta) / l)
        #print(zeta, delta, delta + u / b)

        if zeta >= 2 and delta + u / b >= 2:
            return b
        elif zeta <= delta + u / b:
            #print(np.sqrt(b * delta / l * (1 - delta)))
            return l * min(np.sqrt(b * delta / l * (1 - delta)), 1)
        else:
            return u


def regular_ski_rental(pred: float, eps: float, b: int) -> float:
    '''
    black box learning-to-rent algorithm
    algorithm 3: tau is
    Anand, Ge, Panigrahi
    '''
    tau = np.sqrt(eps)
    if pred >= 1:
        return tau * b  # not sure if floor or ciel np.floor(tau*b)
    else:
        return b


def cal_ski_rental(pred: float, alpha: float, b: int):
    '''
    Algorithm 2: ski-rental with calibrated predictions
    '''
    thresh = (4 + 3 * alpha) / 5
    if pred <= thresh:
        return b
    else:
        ratio = np.sqrt((1 - pred + alpha) / pred + alpha)
        return ratio * b


def get_alg(buy: int, y: int, b: int):
    # buy day is less than number of days skied
    if buy < y:
        # cost is the 1 per day for buy days and then cost of skis
        #return min(buy, b) + b
        return buy + b
        # buy day is after day skied
    elif buy >= y:
        # cost is 1 per day for y days
        return y


def opt_ski_rental(b: int, y: int):
    if y >= b:
        return b
    else:
        return y

def get_CR_df(df: pd.DataFrame, buy_day: str):
    return df.apply(
        lambda row: get_CR(
            buy=row[buy_day],
            y=row['y'],
            b=row['b']
        ),
        axis=1
    )

def get_CR(b: int, y: int, buy: int):
    alg = get_alg(buy=buy, y=y, b=b)
    opt = opt_ski_rental(b=b, y=y)
    return  alg/opt