# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:11:42 2021

@author: gniew
"""
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd  # import pandas
import seaborn as sns   # import seaborn for visualisastion
from matplotlib import pyplot as plt #import matplotlib for plotting data
import pingouin as pt
scale_factor = 0.454/1000 #set scale factor (convert from pixel to mm)

# tu moze byc GUI
df = pd.read_excel("C:\\Users\\gniew\\OneDrive\\Pulpit\\python1_GD.xlsx", sheet_name= "Python", header= 0, engine= "openpyxl") #open data from excel as data frame

df_copy = df #copy data frame to a new one which will be modified


def aligned_for_x_y(rat_id, AP): #Define a function which will align neurons X and Y coordinate to the boundary of NI
    filt_mid_x = (df["id"] == rat_id) & (df["type"]== 7) & (df["AP"] == AP) #Create a filter which will allows to get X value of type 7 [midline of NI] for each brain slices [AP]
    x = df.loc[filt_mid_x, "X"] # get X value of midline and add to new df 
    mean_x = x.mean() # get mean of X value
    filt_x_y = (df["id"] == rat_id) & (df["AP"] == AP) # Create a filter which will allows to get X value of neurons type [types = retrogradelly labeled neurons in NI]
    x1 = df.loc[filt_x_y, "X"] #get X value of neurons and add to new df 
    filt_mid_y = (df["id"] == rat_id) & (df["type"]== 8) & (df["AP"] == AP) #Create a filter which will allows to get Y value of type 8 [lower boundary of NI] for each brain slices [AP]
    y = df.loc[filt_mid_y, "Y"] #get Y value of lower boundry and add to new df 
    mean_y = y.mean() # get mean of Y value
    y1 = df.loc[filt_x_y, "Y"] #get Y value of neurons and add to new df 
    x_column = x1 - mean_x #subtract all data [neuron X value ] from mean x to allign data to midline
    y_column = y1 - mean_y #subtract all data [neuron Y value ] from mean y to allign data to lower boundary 
    finall_A_R = pd.concat([x_column, y_column], axis=1) #create df with two columns [X and Y values of data]
    return finall_A_R #return final df


def objective(x, a, b, c, d, e, f, g):   # szkiel Wielomianu to jesty wielomian 5 stopnia 
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + g


def fit_line(x_val, y_val):
    popt, _ = curve_fit(objective, x_val, y_val)
    a, b, c, d, e, f, g = popt
    x_line = np.arange(min(x_val), max(x_val), 0.01)
    y_line = objective(x_line, a, b, c, d, e, f, g)
    plt.plot(x_line, y_line, '--', color='black')
    plt.gca().set_aspect('equal')
    



rats = pd.unique(df["id"]) #create a list of rats id used in experiment 
slices = pd.unique(df["AP"])  #create a list of brain slices 
types = [1, 2, 5, 6] #list of neurons type 
legends = ["rVTA only", "rVTA only + RLN3", "rVTA + rRMTG", "rVTA + rRMTG + RLN3"]  # create list for legend conntent
slices2 = ["Rostral", "Central", "Caudal"]
colors = ["r", "g", "r", "g"] #this will be used if other colours maps won't fit to figure

for i in range(0, len(rats)): #iterate through rats id 
    for j in range(0, len(slices)): #iterate through brain slices
        rat_slices = aligned_for_x_y(rats[i], slices[j]) # use aligned_for_x_y in for looop
        df_copy.loc[(df_copy["id"] == rats[i]) & (df_copy["AP"] == slices[j]), ["X", "Y"]] = rat_slices #this will substitute rat slices [ product of for loop] to adequate position in df.copy

date_finall = df_copy.loc[:,["X", "Y"]] * scale_factor #(convert from pixel to mm)
date_finall_converted = pd.concat([date_finall.loc[:, "X"], date_finall.loc[:, "Y"]*-1], axis = 1) # merge two data set into one column
date_finall_converted2 = date_finall_converted
date_finall_converted2["type"], date_finall_converted2["AP"], date_finall_converted2["id"] = df_copy["type"], df_copy[ "AP"], df_copy[ "id"]

for i in range(len(slices)):  # iterate through slices index
    plt.figure()  # Create figure for each slices
    for j in range(len(types)):  # iterate through slices index
        plt.style.use('seaborn')  # estetic perpouse 
        plt.scatter(date_finall_converted.loc[(df['AP'] == slices[i]) & (df["type"] == types[j]), "X"], date_finall_converted.loc[(df['AP'] == slices[i]) & (df["type"] == types[j]), "Y"], c = colors[j], cmap = "Set3", alpha= 0.7, label = legends[j])  # plot x  and y + set colour map, alpha and label 
        plt.gca().set_aspect('equal')  # equal axis X and Y 
        plt.title(" {} NI: tracing study".format(slices2[i]))  # Title is added
        plt.ylim(-0.1, 1)  # set Y axis range
        plt.xlim(-1, 1)  # set X axis range
        plt.xlabel("ML axis [mm]")  # X axis title added
        plt.ylabel("DV axis [mm]")  # y axis title added
        plt.legend()  # legend is created
    for k in range(9, 13):
        dupa = date_finall_converted2.loc[(date_finall_converted2['type'] == k) & (date_finall_converted2['AP'] == slices[i]), :]
        fit_line(dupa["X"], dupa["Y"])
     
    
    save_fig_to = "/Users/gniew/OneDrive/Pulpit/tracing_python/figs/NI{}.svg".format(slices2[i])  # save created figure to the file with an adequate title
    plt.savefig(save_fig_to)
plt.show()  # show fig

for i in range(len(slices)):  # iterate through slices index
    plt.figure()
    #sns.kdeplot(date_finall_converted.loc[(df['AP'] == "R") & (df["type"].isin(types)), :], x = "X")
    sns.kdeplot(data = date_finall_converted.loc[(df['AP'] == slices[i]) & (df["type"].isin(types)), :], x = "X", y = "Y", shade=True, cmap="Reds", cbar=True, bw_adjust = 0.4)
    plt.gca().collections[0].set_clim(0,10)
    #plt.style.use('seaborn')  # estetic perpouse 
    #plt.scatter(date_finall_converted.loc[(df['AP'] == slices[i]) & (df["type"] == types[j]), "X"], date_finall_converted.loc[(df['AP'] == slices[i]) & (df["type"] == types[j]), "Y"], c = colors[j], cmap = "Set3", alpha= 0.7, label = legends[j])  # plot x  and y + set colour map, alpha and label 
    #plt.gca().set_aspect('equal')  # equal axis X and Y 
    #plt.title(" {} NI: tracing study".format(slices2[i]))  # Title is added
    plt.ylim(-0.1, 1)  # set Y axis range
    plt.xlim(-1, 1)  # set X axis range
    plt.xlabel("ML axis [mm]")  # X axis title added
    plt.ylabel("DV axis [mm]")  # y axis title added
    #plt.legend()  # legend is created
    for k in range(9, 13):
        dupa = date_finall_converted2.loc[(date_finall_converted2['type'] == k) & (date_finall_converted2['AP'] == slices[i]), :]
        fit_line(dupa["X"], dupa["Y"])
    save_fig_to = "/Users/gniew/OneDrive/Pulpit/tracing_python/figs/NI_density{}.svg".format(slices2[i])  # save created figure to the file with an adequate title
    plt.savefig(save_fig_to)
    plt.show()

df_count_cell = pd.DataFrame(columns = ["rat_id", "cells_number", "side", "slice"], index = range(0,36))
row = 0



for i in rats:
    for j in slices:
        cells_ipsi = len(date_finall_converted2.loc[(date_finall_converted2["X"] >= 0) & (date_finall_converted2["id"] == i) & (date_finall_converted2["AP"] == j) & (date_finall_converted2["type"].isin(types))])
        list_to_add_ipsi = [i, cells_ipsi, "ipsi", j]
        df_count_cell.iloc[row] = list_to_add_ipsi
        row +=1
        cells_ipsi = len(date_finall_converted2.loc[(date_finall_converted2["X"] < 0) & (date_finall_converted2["id"] == i) & (date_finall_converted2["AP"] == j) & (date_finall_converted2["type"].isin(types))])
        list_to_add_contra = [i, cells_ipsi, "contra", j]
        df_count_cell.iloc[row] = list_to_add_contra
        row +=1

df_count_cell_mean = pd.DataFrame(columns = ["mean", "sem", "side", "slice"], index = range(0,6))
row = 0
for j in slices:
    cells_ipsi_mean = round(df_count_cell.loc[(df_count_cell["slice"]== j) & (df_count_cell["side"]== "ipsi"), "cells_number"].mean())
    cells_ipsi_sem = round(df_count_cell.loc[(df_count_cell["slice"]== j) & (df_count_cell["side"]== "ipsi"), "cells_number"].sem(),2)
    list_to_add_ipsi = [cells_ipsi_mean, cells_ipsi_sem, "ipsi", j]
    df_count_cell_mean.iloc[row] = list_to_add_ipsi
    row +=1
    cells_contra_mean = round(df_count_cell.loc[(df_count_cell["slice"]== j) & (df_count_cell["side"]== "contra"), "cells_number"].mean())
    cells_contra_sem = round(df_count_cell.loc[(df_count_cell["slice"]== j) & (df_count_cell["side"]== "contra"), "cells_number"].sem(),2)
    list_to_add_contra = [cells_contra_mean,  cells_contra_sem, "contra", j]
    df_count_cell_mean.iloc[row] = list_to_add_contra
    row +=1

list_brain_side = ["ipsi", "contra"]
x = np.arange(len(list_brain_side))
width = 0.15
multiplier = 0
fig, ax = plt.subplots()
dict_for_bar = {"R": df_count_cell_mean.loc[df_count_cell_mean["slice"] == "R", "mean"].tolist(), "CE": df_count_cell_mean.loc[df_count_cell_mean["slice"] == "CE", "mean"].tolist(), "CA": df_count_cell_mean.loc[df_count_cell_mean["slice"] == "CA", "mean"].tolist()}
sem_list = [[6.2, 5.51], [10.07, 9.98], [12.86, 6.02]]
for key, value in dict_for_bar.items():
    print(value)
    offset = width * multiplier
    rects = ax.bar(x + offset, value, width, label=key, yerr = sem_list[multiplier])
    ax.bar_label(rects, padding=3)
    multiplier += 1
ax.set_ylim(0, 80)
ax.set_xticks(x + width, list_brain_side)
ax.legend(loc='upper left')
save_fig_to = "/Users/gniew/OneDrive/Pulpit/tracing_python/figs/NI_barplot_retro.svg"
plt.savefig(save_fig_to)
plt.show()


sizes = [len(date_finall_converted2.loc[date_finall_converted2["type"].isin([1,5]), "X"]), len(date_finall_converted2.loc[date_finall_converted2["type"].isin([2,6]), "X"])]

fig, ax = plt.subplots()
ax.pie(sizes, autopct='%1.1f%%')
save_fig_to = "/Users/gniew/OneDrive/Pulpit/tracing_python/figs/NI_pie_retro.svg"
plt.savefig(save_fig_to)
plt.show()


list_unknown = ["unknown" for i in range(0,len(date_finall_converted2))]
date_finall_converted2["side"] = list_unknown
date_finall_converted2["side"] = date_finall_converted2["X"].apply(lambda x: "ipsi" if x >= 0 else "contra")

date_finall_converted2.loc[(date_finall_converted2["X"] >= 0)]
### Statystyka
normality_ipsi = df_count_cell.loc[df_count_cell["side"] == "ipsi", "cells_number"].tolist()
normality_contra = df_count_cell.loc[df_count_cell["side"] == "contra", "cells_number"].tolist()
#print("ipsi:", pt.normality(normality_ipsi))
#print("contra:", pt.normality(normality_contra))
df_count_cell.info()
df_count_cell["cells_number"]= df_count_cell["cells_number"].astype('int')
df_count_cell.info()
#pt.kruskal(data=df_count_cell, dv="cells_number", between= 'side') 
#pt.mwu(df_count_cell.loc[df_count_cell["side"] == "ipsi", "cells_number"].tolist(), df_count_cell.loc[df_count_cell["side"] == "contra", "cells_number"].tolist())

normality_ipsi_R = df_count_cell.loc[(df_count_cell["side"] == "ipsi") & (df_count_cell["slice"] == "R"), "cells_number"].tolist() 
normality_ipsi_CE = df_count_cell.loc[(df_count_cell["side"] == "ipsi") & (df_count_cell["slice"] == "CE"), "cells_number"].tolist()
normality_ipsi_CA = df_count_cell.loc[(df_count_cell["side"] == "ipsi") & (df_count_cell["slice"] == "CA"), "cells_number"].tolist()
normality_contra_R = df_count_cell.loc[(df_count_cell["side"] == "contra") & (df_count_cell["slice"] == "R"), "cells_number"].tolist() 
normality_contra_CE = df_count_cell.loc[(df_count_cell["side"] == "contra") & (df_count_cell["slice"] == "CE"), "cells_number"].tolist()
normality_contra_CA = df_count_cell.loc[(df_count_cell["side"] == "contra") & (df_count_cell["slice"] == "CA"), "cells_number"].tolist()


print("ipsi_R:", pt.normality(normality_ipsi_R))
print("ipsi_CE:", pt.normality(normality_ipsi_CE))
print("ipsi_CA:", pt.normality(normality_ipsi_CA))

print("contra_R:", pt.normality(normality_contra_R))
print("contra_CE:", pt.normality(normality_contra_CE))
print("contra_CA:", pt.normality(normality_contra_CA))

print(pt.anova(data= df_count_cell, dv ="cells_number", between = ["side", "slice"], detailed = True))