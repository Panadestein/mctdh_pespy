""" This module parses the PES section of a MCTDH's operator file
    to generate the corresponding SOP representaion. It can also be
    used to generate geometries on different multidimensional grids
    or with a Markov Chain Monte Carlo process"""
import itertools
import numpy as np
import scipy.constants as sc
import fucn_mctdh as fn

# Convertion factors

AU_ANG = sc.physical_constants['atomic unit of length'][0] * 1e10
AU_CMINV = sc.physical_constants['hartree-inverse meter relationship'][0] * .01
DEG_RAD = np.pi / 180

# Equilibrium parameters for trans isomer (Rosmus et al.)

R_2_TRANS0 = 1.170 / AU_ANG
R_3_TRANS0 = 1.426 / AU_ANG
R_1_TRANS0 = 0.964 / AU_ANG
T_2_TRANS0 = 110.70 * DEG_RAD
T_1_TRANS0 = 101.90 * DEG_RAD
P_1_TRANS0 = 180.0 * DEG_RAD
X0 = [R_2_TRANS0, R_3_TRANS0, R_1_TRANS0, T_2_TRANS0, T_1_TRANS0,
      P_1_TRANS0]

# Define grid bounds (Rosmus et al.)

R_2_MIN, R_2_MAX = 1.95, 2.65
R_3_MIN, R_3_MAX = 2.20, 3.60
R_1_MIN, R_1_MAX = 1.50, 2.50
T_2_MIN, T_2_MAX = 2.27838076, 1.67096375
T_1_MIN, T_1_MAX = 2.27838076, 1.31811607
P_1_MIN, P_1_MAX = 0.00, 3.14

# Generation of geometries in 1D grids

NGEOS = 5
COORDS = {'r_2': np.linspace(R_2_MIN, R_2_MAX, num=NGEOS),
          'r_3': np.linspace(R_3_MIN, R_3_MAX, num=NGEOS),
          'r_1': np.linspace(R_1_MIN, R_1_MAX, num=NGEOS),
          't_2': np.linspace(T_2_MIN, T_2_MAX, num=NGEOS),
          't_1': np.linspace(T_1_MIN, T_1_MAX, num=NGEOS),
          'p_1': np.linspace(P_1_MIN, P_1_MAX, num=NGEOS)}

# Generation of geometries in 2D grids

COORD_2D = []
for subset in itertools.combinations(COORDS.keys(), 2):
    rx_lab, ry_lab = subset
    rx, ry = COORDS[rx_lab], COORDS[ry_lab]
    r1, r2 = np.meshgrid(rx[::2], ry[::2])
    coord_r = np.vstack([r1.flatten(), r2.flatten()]).T
    COORD_2D.append(coord_r)
COORD_2D = np.array(COORD_2D)

# Parse PES section of MCTDH operator file

with open('hono_mctdh', 'r') as inf:
    for idx, line in enumerate(inf):
        if "HAMILTONIAN-SECTION" in line:
            h_beg = idx + 4
        elif "end-hamiltonian-section" in line:
            h_end = idx - 1
            break
with open('hono_mctdh', 'r') as inf:
    DATA_POT = np.genfromtxt(inf, dtype='str', skip_header=h_beg,
                             max_rows=(h_end - h_beg), delimiter="|")

# Compute PES for a given geometry


def func_ev(dof):
    """Dictionary comprehension to evaluate functions inside the
    operator file on the grid points"""
    return {key: value(dof) for (key, value) in fn.FUNCIONES.items()}


def hono_pes(args):
    """Computes the values as a Sum of Products of terms present in the
    PES section of the MCTDH operator file. Here the use of eval() built-in
    function if the func_ev(dof) global namespace is something to improve"""
    r_2, r_3, r_1 = args[0], args[1], = args[2]
    t_2, t_1 = np.cos(args[3]), np.cos(args[4])
    p_1 = args[5]
    pot = 0.0
    for elem in DATA_POT:
        coeff = float(elem[0].replace('d', 'E'))
        term_1 = eval(elem[1].replace('^', '**'), func_ev(r_2))
        term_2 = eval(elem[2].replace('^', '**'), func_ev(r_3))
        term_3 = eval(elem[3].replace('^', '**'), func_ev(r_1))
        term_4 = eval(elem[4].replace('^', '**'), func_ev(t_2))
        term_5 = eval(elem[5].replace('^', '**'), func_ev(t_1))
        term_6 = eval(elem[6].replace('^', '**'), func_ev(p_1))
        pot += coeff * term_1 * term_2 * term_3 * term_4 * term_5 * term_6
    return pot * AU_CMINV


# Generate geometries with the Metropolis–Hastings algorithm

def mcmc(x_vect, ngeos):
    """A simple implementation of MCMC"""
    seti = []
    kbt = 2000
    count = 0
    while count < ngeos:
        x_new = x_vect + np.random.uniform(-.1, .1, len(x_vect))
        delta_e = hono_pes(x_new) - hono_pes(x_vect)
        if delta_e <= 0:
            seti.append(np.append(x_new, hono_pes(x_new)))
            x_vect = x_new
            count += 1
        else:
            p_dist = np.exp(-delta_e / kbt)
            u_rnd = np.random.uniform(0, 1)
            if u_rnd <= p_dist:
                seti.append(np.append(x_new, hono_pes(x_new)))
                x_vect = x_new
                count += 1
    return seti


GEOS_MCMC = mcmc(X0, 10)
GEOS_MCMC = np.array(GEOS_MCMC)
np.savetxt('geos_mcmc', GEOS_MCMC)
