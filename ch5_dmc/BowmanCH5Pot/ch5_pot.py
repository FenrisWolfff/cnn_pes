from libCH5pot import mycalcpot
import numpy as np

def rjd_ch5(cds):
    v = mycalcpot(cds,len(cds))
    return v

if __name__ == '__main__':
    a = np.random.random((100,6,3))
    print(rjd_ch5(a))