# -*- coding: utf-8 -*-
"""
Half life and decay constant calculation program.
A code to find the half life and decay constants for Rubidium and Strontium in a decay chain 
starting with 10^-6 mols of Strontium.
Joe Tarnoky 11/12/2019
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import sys

lambdas = [0.0005, 0.005] #initial guesses of decay constants for Rubidium and Strontium respectively.
Ns_initial = 6.023 * 10**17 #initial number of Strontium nuclei

def Number_Rb(t, lambdas):
    """Calculates how the number of Rubidium Nuclei as a function of time.
    t(float)
    lamdas(array)"""
    
    Ls = lambdas[1]  #assigns value to decay constant of Strontium
    Lr = lambdas[0]  #assigns value to decay constant of Rubidium
    number_Rb = Ns_initial * (Ls / (Lr - Ls)) * (np.exp(-Ls * t) - np.exp(-Lr * t)) 
    
    return number_Rb
    
    
def activity_Rb(t, lambdas):
    """Calculatesthe activity of Rubidium as a function of time.
    t(float)
    lambdas(array)"""
    
    Lr = lambdas[0]
    Activity_Rb = (Lr * Number_Rb(t, lambdas)) / 10**(12) #converted into TBq
    return Activity_Rb
    
    
    
def read_data_function():
    """Reads in data files and removes lines that contain errors or invalid numbers. Causes code to stop and asks user to input different files 
    if the inputted files do not exist."""
    
    try:
        input_file_1 = np.genfromtxt('Nuclear_data_1.csv', delimiter = ',')
        input_file_2 = np.genfromtxt('Nuclear_data_2.csv', delimiter = ',')
    except:
        print('Please check files within read_data_function exist and run the code again.')
        sys.exit()
    
    
    stacked_data = np.vstack((input_file_1, input_file_2))
    full_data = np.zeros((0, 3))
    
    for i in range(len(stacked_data)):
            
        if 0 < stacked_data[i,0] and 0 < stacked_data[i,1] and 0 < stacked_data[i,2]:
            temp = np.array([float(stacked_data[i,0]) * 3600, float(stacked_data[i,1]), float(stacked_data[i,2])]) # '*3600' is to convert to seconds
            full_data = np.vstack((full_data, temp))
    
    sorted_data = full_data[np.argsort(full_data[:,0]),:]
    
    return sorted_data

def outlier_identification():
    """Removes outliers from the data set. Identifies an outlier if its distance from the
    theoretical fit is greater than 3 times its associated error."""
    
    clean_data = np.zeros((0, 3))
    for i in range(len(sorted_data)):
        if abs(sorted_data[i,1] - activity_Rb(sorted_data[i,0], lambdas)) < (3 * sorted_data[i,2]):
            temp = np.array((sorted_data[i,0], sorted_data[i,1], sorted_data[i,2]))
            clean_data = np.vstack((clean_data, temp))
        else:
            pass
    return clean_data
        

def activity_vs_time_plot(clean_data):
    """Produces a plot of Activity vs Time by plotting the data points and by plotting the theoretical fit
    against the time data points, also saves the plot.
    clean_data(array)"""
    
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.set_title('Activity vs Time', fontsize = 16, fontname = 'Times New Roman', color = 'black')
    axes.grid(True, color = 'grey', dashes = [8,8])
    axes.set_xlabel('Time (s)', fontsize = 14, fontname = 'Times New Roman', color = 'black')
    axes.set_ylabel('Activity (TBq)', fontsize = 14, fontname = 'Times New Roman', color = 'black')
    axes.errorbar(clean_data[:,0], clean_data[:,1], clean_data[:,2], linestyle = 'none', 
                      marker = 'o', markersize = 3, color = 'navy', ecolor = 'teal',  capsize = 3, markeredgewidth = 1,  
                          label = 'observed data')
    axes.plot(clean_data[:,0], activity_Rb(clean_data[:,0], lambdas), color = 'indigo', label = 'theoretical fit')
    axes.legend(loc='best')
    plt.savefig('Activity vs Time plot.png', dpi = 300)
    plt.show()
              

def chi_squared(lambdas):
    """Calculates the chi squared value.
    lambdas(array)"""
    
    calculated_activity = activity_Rb(sorted_data[:,0], lambdas)
    chi_squared = np.sum(((sorted_data[:,1] - calculated_activity) / sorted_data[:,2])**2)
    
    return chi_squared 

def reduced_chi_squared(lambdas):
    """Finds the reduced chi squared value from the chi squared value.
    lambdas(array)"""

    red_chi_squared = chi_squared(lambdas) / (len(sorted_data) - 2)
    
    print('The reduced chi squared value is {0:3.2f}' .format(red_chi_squared))
    
    return red_chi_squared 

def minimise_chi_squared():
    """Minimises the chi squared value by varying both decay constants until chi squared
    is at a minimum."""
    
    chi_squared_minimisation = fmin(chi_squared, lambdas, disp = 0)
    
    lambdas[1] = chi_squared_minimisation[1]  
    lambdas[0] = chi_squared_minimisation[0]
    chi_min = chi_squared(chi_squared_minimisation)
    
    return chi_min
    
    
def half_lives(lambdas):
    """Calculates the half lives of Rubidium and Strontium
    lambdas(array)"""
    
    half_life_Rb = np.log(2) / (lambdas[0] * 60)
    half_life_Sr = np.log(2) / (lambdas[1] * 60)
    
    return half_life_Rb, half_life_Sr
        
def chi_squared_1(lambdas):
    """Calculates the difference between the chi squared function and the
    minimised chi squared function - 1.
    lambdas(array)"""
    
    chi_min = minimise_chi_squared()
    x = np.abs(chi_squared(lambdas) - chi_min - 1)
    return x

def uncertainties(half_life_Rb, half_life_Sr):
    """Minimises the the difference between the chi squared function and the
    minimised chi squared function - 1 and used the new values of lambdas associated with
    this minimisation to calculate errors for the decay constants. Also propogates these 
    errors to calculate errors on the half lives.
    half_life_Rb[float]
    half_life_Sr[float]"""
    
    chi_minimization_error = fmin(chi_squared_1, lambdas, disp = 0)
    
    error_Lr = np.abs(chi_minimization_error[0] - lambdas[0])
    error_Ls = np.abs(chi_minimization_error[1] - lambdas[1])
    
    error_half_life_Rb = (error_Lr * half_life_Rb) / lambdas[0]
    error_half_life_Sr = (error_Ls * half_life_Sr) / lambdas[1]
    
    return error_Ls, error_Lr, error_half_life_Rb, error_half_life_Sr
    

def contours():
    """Makes a contour plot displaying how chi squared varies with the 2 decay constants."""
    
    x = np.linspace(Ls - (Ls * 0.1), Ls + (Ls * 0.1), 200)
    y = np.linspace(Lr - (Lr * 0.1), Lr + (Lr * 0.1), 200)
    X, Y = np.meshgrid(x, y)
    Z=np.zeros((len(x),len(y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            coords = np.array([X[i][j],Y[i][j]])
            Z[i][j] = chi_squared(coords) 
    levels = (chi_min + 2.3, chi_min + 5.99, chi_min + 9.21)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    contour = axes.contour(X, Y, Z, levels)
    axes.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    contour_filled = plt.contourf(X, Y, Z, 20, cmap = 'plasma_r')

    axes.set_title('Chi Squared Contour Plot', fontsize = 16, fontname = 'Times New Roman', color = 'black')
    axes.set_xlabel('Strontium decay constant ($s^{-1}$)', fontsize = 14, fontname = 'Times New Roman', color = 'black')
    axes.set_ylabel('Rubidium decay constant ($s^{-1}$)', fontsize = 14, fontname = 'Times New Roman', color = 'black')
    plt.savefig('Chi Squared Contour Plot', dpi = 300)
    plt.show()
    
    
    return X, Y, Z
    
#===============================main code=====================================#        


sorted_data = read_data_function()  

chi = minimise_chi_squared() 
 
sorted_data = outlier_identification() 

activity_vs_time_plot(sorted_data) 

chi_min = minimise_chi_squared()

reduced_chi_squared(lambdas) 

half_life_Rb, half_life_Sr = half_lives(lambdas)

error_Ls, error_Lr, error_half_life_Rb, error_half_life_Sr = uncertainties(half_life_Rb, half_life_Sr)

print('The decay constant of Strontium is {0:3.5f} \u00b1 {1:3.5f} s^-1' .format(lambdas[1], error_Ls))
print('The decay constant of Rubidium is {0:3.6f} \u00b1 {1:3.6f} s^-1' .format(lambdas[0], error_Lr))
print('The half life of Rubidium is {0:3.1f} \u00b1 {1:3.1f} mins' .format(half_life_Rb, error_half_life_Rb))
print('The half life of Strontium is {0:3.2f} \u00b1 {1:3.2f} mins' .format(half_life_Sr, error_half_life_Sr))
    
Ls = lambdas[1]  #updates lambda values to ensure contour plot uses calculated values. 
Lr = lambdas[0]  #all previous updating of these values takes place inside other functions, which is why this update is necessary.

X, Y, Z = contours()