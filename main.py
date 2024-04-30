import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



def halton_sequence(count, base=0):
    sequence = []
    prime = [2, 3][base]  # First prime number
    while len(sequence) < count:
        value = 0
        f = 1 / prime
        index = len(sequence) + 1
        while index > 0:
            value += f * (index % prime)
            index //= prime
            f /= prime
        sequence.append(value)
    return sequence[:count]


def Box_Muller(M=10 ** 4, type='MC'):
    if type == 'MC':
        M = int(M)
        N = M // 2 if M % 2 else (M + 1) // 2
        U_1 = np.random.uniform(size=N)
        U_2 = np.random.uniform(size=N)
        Theta = 2 * np.pi * U_1
        R = np.sqrt(-2 * np.log(U_2))
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Norm = np.concatenate((X, Y))[:M]
        return (Norm)
    elif type == 'QMC':
        if M % 2 == 0:
            M = int(M)
            U_1 = np.array(halton_sequence(M, 0))
            U_2 = np.array(halton_sequence(M, 1))
            Theta = 2 * np.pi * U_1
            R = np.sqrt(-2 * np.log(U_2))
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)
        else:
            M = int((M + 1) / 2)
            U_1 = np.array(halton_sequence(M, 0))
            U_2 = np.array(halton_sequence(M, 1))
            Theta = 2 * np.pi * U_1
            R = np.sqrt(-2 * np.log(U_2))
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)
            Y = Y[:-1]
        Norm = np.concatenate((X, Y))
        return (Norm)

def box_muller(u):
    '''funkcja generujaca M zmiennych normalnych'''
    length = len(u)
    if length % 2 != 0:
        raise ValueError("Length of input array must be even")
    midpoint = length // 2
    u_1 = u[:midpoint]
    u_2 = u[midpoint:]
    theta = 2 * np.pi * u_1
    r = np.sqrt(-2 * np.log(u_2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    norm = np.concatenate((x,y))
    return(norm)


def CallMC(S0=50, r=0.02, sigma=0.3, T=0.5, K=50, M=10 ** 4, conf='TRUE'):
    # Monte Carlo
    if conf == 'TRUE':
        t = T
        N = Box_Muller(M)
        St_MC_plus = S0 * np.exp((r - sigma ** 2 / 2) * t + sigma * N * np.sqrt(t))
        St_MC_minus = S0 * np.exp((r - sigma ** 2 / 2) * t + sigma * - N * np.sqrt(t))
        V_MC_plus = np.maximum(St_MC_plus - K, 0)
        V_MC_minus = np.maximum(St_MC_minus - K, 0)
        payoff_MC = np.exp(-r * t) * (V_MC_plus + V_MC_minus) / 2
        price_MC = np.mean(payoff_MC)
        confidence_interval_up = price_MC + 1.96 * np.std(payoff_MC) / np.sqrt(M)
        confidence_interval_down = price_MC - 1.96 * np.std(payoff_MC) / np.sqrt(M)
        # Quasi Monte Carlo
        N = Box_Muller(M, type='QMC')
        St_QMC = S0 * np.exp((r - sigma ** 2 / 2) * t + sigma * N * np.sqrt(t))
        payoff_QMC = np.maximum(St_QMC - K, 0)
        price_QMC = np.exp(-r * t) * np.mean(payoff_QMC)

        return (price_MC, price_QMC, confidence_interval_up, confidence_interval_down)
    else:
        t = T
        N = Box_Muller(M)
        St_MC = S0 * np.exp((r - sigma ** 2 / 2) * t + sigma * N * np.sqrt(t))
        payoff_MC = np.maximum(St_MC - K, 0)
        price_MC = np.exp(-r * t) * np.mean(payoff_MC)
        # QMC
        N = Box_Muller(M, type='QMC')
        St_QMC = S0 * np.exp((r - sigma ** 2 / 2) * t + sigma * N * np.sqrt(t))
        payoff_QMC = np.maximum(St_QMC - K, 0)
        price_QMC = np.exp(-r * t) * np.mean(payoff_QMC)

        return (price_MC, price_QMC)

def call_mc(S0=50, r=0.02, sigma=0.3, T=0.5, K=50, M=10 ** 4):
    u = np.random.uniform(size=M)
    z = box_muller(u)
    a = sigma * z * np.sqrt(T)
    b = (r - sigma ** 2 / 2) * T
    st_plus = S0 * np.exp(b + a)
    st_minus = S0 * np.exp(b + -a)
    v_plus = np.maximum(st_plus - K, 0)
    v_minus = np.maximum(st_minus - K, 0)
    payoff = np.exp(-r * T) * (v_plus + v_minus) / 2
    price = np.mean(payoff)
    return(price)

def call_qmc(S0=50, r=0.02, sigma=0.3, T=0.5, K=50, M=10 ** 4):
    N = M // 2
    u_1 = halton_sequence(N,0)
    #u_1 = np.array(u_1)
    u_2 = halton_sequence(N, base=1)
    #u_2 = np.array(u_2)
    u = np.concatenate([u_1,u_2])
    z = box_muller(u)
    a = sigma * z * np.sqrt(T)
    b = (r - sigma ** 2 / 2) * T
    st = S0 * np.exp(b + a)
    v = np.maximum(st - K, 0)
    price = np.exp(-r * T) * np.mean(v)
    return (price)

def BSC(S0=50, r=0.02, sigma=0.3, T=0.5, K=50):
    """
    Calculate the Black-Scholes model price of a call option.

    Parameters:
    S0 (float): Current price of the underlying asset.
    r (float): Annual risk-free interest rate.
    sigma (float): Annual volatility of the underlying asset.
    T (float): Time to expiration of the option (in years).
    K (float): Strike price of the option.

    Returns:
    float: Black-Scholes model price of the call option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

if __name__ == "__main__":
    parameters = {}

    with open('cp2_data.txt', 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ')
            parameters[key] = float(value)
    price_MC, price_QMC, CI_UP, CI_DOWN = CallMC(parameters['S0'], parameters['r'], parameters['sigma'],
                                                 parameters['T'],
                                                 parameters['K'], parameters['M'])
    bsc = BSC(parameters['S0'], parameters['r'], parameters['sigma'], parameters['T'], parameters['K'])
    remc = abs((price_MC - bsc) / bsc)
    reqmc = remc = abs((price_QMC - bsc) / bsc)
    print("Wycena opcji metodą Monte Carlo", price_MC, 'Przedział ufności górny', CI_UP, 'Przedział ufności dolny',
          CI_DOWN)
    print('Bład względny wyceny metodą Monte Carlo', remc)
    print('Wycena opcji metodą Quasi Monte Carlo', price_QMC, )
    print('Bład względny wyceny metodą Quasi Monte Carlo', reqmc)



def sym_MC(K, N=100, start=1000, interv=100):
    cena_BS = BSC(K=K)
    sim_number = np.arange(start, start + N * interv, interv)
    results_MC = np.empty_like(sim_number, dtype=float)
    results_QMC = np.empty_like(sim_number, dtype=float)
    bound_error_MC = 1 / np.sqrt(sim_number)
    bound_error_QMC = np.log(sim_number) / sim_number
    for i, M in enumerate(sim_number):
        place_holder = CallMC(M=M, K=K, conf='')
        difference_MC = abs((place_holder[0] - cena_BS) / cena_BS)
        difference_QMC = abs((place_holder[1] - cena_BS) / cena_BS)
        results_MC[i] = difference_MC
        results_QMC[i] = difference_QMC
    data = pd.DataFrame(
        {'MC': results_MC, 'QMC': results_QMC, 'BEMC': bound_error_MC, 'BEQMC': bound_error_QMC, 'SIM': sim_number})
    return (data)

def sym_mc(K, start=1000, step=100, step_size=100):
    cena_BS = BSC(K=K)
    sim_number = np.arange(start, start + step * step_size, step_size)
    results_MC = np.empty_like(sim_number, dtype=float)
    bound_error_MC = 1 / np.sqrt(sim_number) * 1
    for i, M in enumerate(sim_number):
        c = call_mc(M=M, K=K)
        difference_MC = abs((c - cena_BS))
        results_MC[i] = difference_MC
    data = pd.DataFrame(
        {'MC': results_MC, 'BEMC': bound_error_MC, 'SIM': sim_number})
    return (data)

def sym_qmc(K, start=1000, step=100, step_size=100):
    cena_BS = BSC(K=K)
    sim_number = np.arange(start, start + step * step_size, step_size)
    results_QMC = np.empty_like(sim_number, dtype=float)
    bound_error_QMC = np.log(sim_number) / sim_number * 5
    for i, M in enumerate(sim_number):
        c = call_qmc(M=M, K=K)
        difference_MC = abs((c - cena_BS))
        results_QMC[i] = difference_MC
    data = pd.DataFrame(
        {'MC': results_QMC, 'BEMC': bound_error_QMC, 'SIM': sim_number})
    return (data)

#QMC
"""
data = sym_qmc(K=80, start=1000, step=100, step_size=1000)

# Plot MC as a line
plt.scatter(data['SIM'], data['MC'], label='Error Quasi Monte Carlo', alpha = 0.8)
plt.plot(data['SIM'], data['BEMC'], label='Bound of error Quasi Monte Carlo', c='r')

# Add labels and title
plt.xlabel('Number of simulations')
plt.ylabel('Quasi Monte Carlo')
plt.title('Quasi Monte Carlo error vs theoretical bound (OTM)')
plt.legend()

# Show the plots
plt.show()
"""

#MC
#"""

data = sym_mc(K=80, start=1000, step=100, step_size=1000)

# Plot MC as a line
plt.scatter(data['SIM'], data['MC'], label='Error Monte Carlo', alpha = 0.8)
plt.plot(data['SIM'], data['BEMC'], label='Bound of error Monte Carlo', c='r')

# Add labels and title
plt.xlabel('Number of simulations')
plt.ylabel('Monte Carlo')
plt.title('Monte Carlo error vs theoretical bound (OTM)')
plt.legend()

# Show the plot
plt.show()
#"""
#MC - ATM = 6
#MC - ITM = 4
#MC - OTM = 1