""" Contains lambda function definitions to support parsing from
    MCTDH operator file"""
import numpy as np


def exp1(x, p1, p2, r):
    return (1 - np.exp(p1 * (x - p2))) ** r


def acos(x, p1, p2, p3, r):
    return (np.arccos(p1 * x - p2) - p3) ** r


def qs(x, p, r):
    return (np.sqrt(p - x**2)) ** r


def gauss(x, p1, p2, r):
    return np.exp(-p1 * (x - p2) ** 2) ** r


def tgauss(x, p1, p2, r):
    return np.exp(-p1 * (np.arccos(x) - p2) ** 2) ** r


def cos(x, p1, p2, r):
    return (np.cos(p1 * (x - p2))) ** r


Q20 = 2.696732586
Q30 = 1.822912197
Q10 = 2.213326419
Q11 = 1.8653
TH20 = 1.777642018
TH10 = 1.9315017
p1 = np.pi / 2.0
p2 = 3.0 * np.pi / 2.0

FUNCIONES = {
    "q1": (lambda x: exp1(x, -0.70, Q10, 1)),
    "q2": (lambda x: exp1(x, -0.70, Q20, 1)),
    "q3": (lambda x: exp1(x, -0.70, Q30, 1)),
    "t1": (lambda x: acos(x, 1.0, 0.0, TH20, 1)),
    "t2": (lambda x: acos(x, 1.0, 0.0, TH10, 1)),
    "qs1": (lambda x: qs(x, 1.0, 1)),
    "g1": (lambda x: gauss(x, 1.0, Q10, 1)),
    "g2": (lambda x: gauss(x, 1.0 / 9.0, Q20, 1)),
    "g3": (lambda x: gauss(x, 0.25, Q30, 1)),
    "g4": (lambda x: tgauss(x, 0.25, TH10, 1)),
    "g5": (lambda x: tgauss(x, 0.25, TH20, 1)),
    "c1": (lambda x: gauss(x, -4.0, Q10, 1)),
    "c2": (lambda x: gauss(x, -1.0, Q20, 1)),
    "c3": (lambda x: gauss(x, -4.0, Q30, 1)),
    "c4": (lambda x: tgauss(x, -4.0, TH10, 1)),
    "c5": (lambda x: tgauss(x, -4.0, TH20, 1)),
    "g0pi": (lambda x: gauss(x, 2.0, 0.0, 1)),
    "g1pi": (lambda x: gauss(x, 2.0, np.pi, 1)),
    "g2pi": (lambda x: gauss(x, 2.0, 2.0 * np.pi, 1)),
    "cos": (lambda x: cos(x, 1.0, 0.0, 1)),
    "cos2": (lambda x: cos(x, 2.0, 0.0, 1)),
    "cos3": (lambda x: cos(x, 3.0, 0.0, 1)),
    "cos4": (lambda x: cos(x, 4.0, 0.0, 1)),
}
