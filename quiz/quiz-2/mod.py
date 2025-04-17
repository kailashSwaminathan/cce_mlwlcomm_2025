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
            symbols = symbols.astype(complex)
        case 'qpsk':
            bits = np.random.randint(0, 4, numsamples)
            symbols = np.exp(1j * ( (np.pi/4) + (np.pi/2)*bits ))
        case '16qam':
            # generate random 16-QAM symbols
            symbols = (2 * np.random.randint(0,4,numsamples) - 3) + 1j * (2 * np.random.randint(0,4,numsamples) - 3)
            # normalize power
            symbols = symbols / np.sqrt(10)
        case '8psk':
            bits = np.random.randint(0,8,numsamples)
            symbols = np.exp(1j * ( (2*np.pi/8) * bits) )
        case _:
            print(f'{mod} Not implemented. Possible values (bpsk | qpsk | 16qam)')
            return None
            
    return symbols
    
def create_modulation_data(numsamples, numsymbols, modlist, add_noise=1, noisevar=1, noisemean=0, snrdb=0):
    """
    """
    mod_data = np.empty((0,numsymbols))
    for indx,modtype in enumerate(modlist):
        sigdata = generate_samples((numsamples * numsymbols), modtype)
        if add_noise:
            noise = generate_awgn((numsamples * numsymbols), mean=noisemean, variance=noisevar, snrdb=snrdb)
            sigdata += noise
        sigdata = sigdata.reshape(numsamples,numsymbols)
        mod_data = np.concatenate([mod_data, sigdata])
    return mod_data    


def _test_8psk(numsamples):
    """
    """
    data = generate_samples(numsamples, '8psk')
    return data
    

if __name__ == "__main__":
    numsamples = 1000
    symbols = _test_8psk(numsamples)
    print(f'Power: {np.var(symbols)}')
    plt.plot(np.real(symbols), np.imag(symbols),'g*')
    plt.show()
