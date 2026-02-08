"""Profit and loss simulation for decision policies."""

import numpy as np
import pandas as pd


def simulate_profit(
    y_true,
    y_pred_proba,
    threshold=0.5,
    avg_loan_amount=200000,
    interest_rate=0.18,
    loss_given_default=0.7,
    operational_cost=2000
):

    decisions = (y_pred_proba < threshold).astype(int)

    profits = []

    for i in range(len(y_true)):

        if decisions[i] == 0:
            profits.append(0)
            continue

        if y_true.iloc[i] == 0:
            profit = avg_loan_amount * interest_rate - operational_cost
        else:
            profit = -avg_loan_amount * loss_given_default - operational_cost

        profits.append(profit)

    total_profit = np.sum(profits)
    avg_profit = np.mean(profits)

    return {
        "total_profit": total_profit,
        "avg_profit_per_customer": avg_profit,
        "approval_rate": decisions.mean()
    }
