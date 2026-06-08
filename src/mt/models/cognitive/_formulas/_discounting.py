"""Temporal discounting formulas."""

from __future__ import annotations


def hyperbolic_discounting_logits(rewards, delays, beta, discount_rate):
    return beta * rewards.float() / (1 + discount_rate * delays.float())
