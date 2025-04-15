import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_awgn(numsamples, mean=0, variance=1, snrdb=0, spower=1):
    """
    """
    snr_linear = 10 ** (snrdb/10)
    npower = spower/snr_linear
    noise = np.sqrt(npower) * (np.random.normal(loc=mean, scale=np.sqrt(variance), size=numsamples) + \
                                 1j * np.random.normal(loc=mean,scale=np.sqrt(variance),size=numsamples))
    return noise
    

def generate_samples(numsamples, mod, a=None):
    """ Generate samples with a specified modulation
        scheme.
    """
    match(mod):
        case 'bpsk':
            bits = np.random.randint(0,2,numsamples)
            symbols = (2*bits - 1)
        case 'qpsk':
            bits = np.random.randint(0, 4, numsamples)
            symbols = np.exp(1j * ( (np.pi/4) + (np.pi/2)*bits ))
        case '16qam':
            # mappingTable = [
                # -3-3j, -3-1j, -3+3j, -3+1j,
                # -1-3j, -1-1j, -1+3j, -1+1j,
                 # 3-3j,  3-1j,  3+3j,  3+1j,
                 # 1-3j,  1-1j,  1+3j,  1+1j 
            # ]
            # bits = np.random.randint(0,16, numsamples)
            # symbols = [(1/np.sqrt(10))*mappingTable[i] for i in bits]
            # generate random 16-QAM symbols
            symbols = (2 * np.random.randint(0,4,numsamples) - 3) + 1j * (2 * np.random.randint(0,4,numsamples))
            # normalize power
            symbols = symbols / np.sqrt(10)
        case _:
            print(f'{mod} Not implemented. Possible values (bpsk | qpsk | 16qam)')
            return None
            
    return symbols



if __name__ == "__main__":
    numsamples = 1000
    symbols = generate_samples(numsamples,'bpsk')
    noise = generate_awgn(numsamples,variance=0.1,snrdb=5)
    symbols = symbols.astype(complex) + noise
    print(f'Power: {np.var(symbols)}')
    plt.plot(np.real(symbols), np.imag(symbols),'g*')
    plt.show()
