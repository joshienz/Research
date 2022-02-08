# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:35:16 2021

@author: NICHOLSJA18
"""
import matplotlib.pyplot as plt
import numpy as np



# Adjust for screen resolution
if plt.get_backend() == 'Qt5Agg':
    import sys
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = 0.8*qApp.desktop().physicalDpiX()



def main():


    
    
    r1, eigA, eigB, eigC, eigD = np.loadtxt('newBasis.txt',delimiter='\t',unpack=True)
    r2, eig1, eig2, eig3, eig4 = np.loadtxt('newBasis2.txt',delimiter='\t',unpack=True)
    #r3, sum3 = np.loadtxt('res_l9.txt',delimiter='\t',unpack=True)
    #r4, sum4 = np.loadtxt('res_l5.txt',delimiter='\t',unpack=True)
    #r5, sum5 = np.loadtxt('res_l3.txt',delimiter='\t',unpack=True)
    """
    for i,eig in enumerate(eigA):
        if(eig<0):
            eigA[i] = eig+np.pi
    for i,eig in enumerate(sum2):
        if(eig<0):
            sum2[i] = eig+np.pi
    """
    fig, ax = plt.subplots()
    ax.plot(r1, eigA)
    #ax.plot(r1, eigB)
    #ax.plot(r1, eigC)
    #ax.plot(r1, eigD)
    #ax.plot(r2, eig1)
    #ax.plot(r2, eig2)
    #ax.plot(r2, eig3)
    #ax.plot(r2, eig4)
    #ax.plot(r3, sum3)
    #ax.plot(r4, sum4)
    #ax.plot(r5, sum5)
    #ax.set(xlabel='phase shift', ylabel='r',
           #title='R Vs phase shift')
    plt.show()
    plt.tight_layout()
    plt.pause(2.5)
    
    """
    r = []
    sum0 = []
    val = float(input("where does the first graph end? enter k value\n"))
    for i in range(len(r1)):
        
        if r1[i]<=val:
            r.append(r2[i])
            sum0.append(sum2[i])
        else:
            r.append(r1[i])
            sum0.append(sum1[i])
    
    fig, ax = plt.subplots()
    ax.plot(r, sum0)
    ax.set(xlabel='phase shift', ylabel='r',
           title='R Vs phase shift')
    plt.show()
    plt.tight_layout()

    np.savetxt('res_l13.txt',(r,sum0))
    
    with open('res_l13.txt','w') as txtFile:
        for i in range(len(r)):
            txtFile.write("{}".format(r[i]))
            txtFile.write("\t{}\n".format(sum0[i]))

    """


    
    
    
if __name__ == '__main__':
    #start_time = time.time()
    main()
    #end_time = time.time()
    #print('Elapsed time: {:.2f} seconds'.format(end_time-start_time))