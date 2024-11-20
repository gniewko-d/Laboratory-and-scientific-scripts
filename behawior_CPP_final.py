# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:27:24 2023

@author: gniew
"""

import numpy as np # for matrix calculation
from scipy.optimize import curve_fit
import pandas as pd  # import pandas
import seaborn as sns   # import seaborn for visualisastion
from matplotlib import pyplot as plt #import matplotlib for plotting data
import math as m # calculation
import statistics as stat # easy statistic
from scipy.spatial import distance # euclidean distance
import tkinter as tk # creat root widget
from tkinter import filedialog # open file
import pingouin as pg # not important yet.....
import easygui
from scipy.signal import savgol_filter
import math



def euclides(data1, data2, column: str): # creat function that will calculate euclides distance 
    ser = []  # creat empty list 
    for i in range(0, len(data1)-1):  #  iterate thorugh data witout last frame 
        
        point_1 = [data1[i], data2[i]]
        point_2 = [data1[i+1], data2[i+1]]
        
        #x = [data1[i+1] - data1[i]]  # get subraction result for x cordinate
        #y = data2[i+1] - data2[i]  # get subraction result for y cordinate
        #e = distance.euclidean(x, y)  #  calculate euclides distance
        e = math.dist(point_1, point_2)
        ser.append(e)       # append next distance 
    ser.insert(0,0) # first frame no distance moved 
    serx = pd.Series(ser, name = column)  # creat pd series 
    return serx   #return pd series 

# constans to set
cpp_cm = 25 # cpp inside lenght of longer arm of C arena IN REAL WORLD 
cpp_pix_left = 346
cpp_pix_right = 531
cpp_pix = cpp_pix_right - cpp_pix_left # cpp inside lenght of longer arm of C arena IN DIGITAL WORLD

pix_to_cm = cpp_cm/cpp_pix
cm_to_pix = 1/pix_to_cm



starting_frame =510# set this after watching each video from experiment 
probability_treshold = 0.9  # depend on how good is deep learning algorithm
distance_treshold = 30  # not important yet  
i = 0
Ax = 395  # get this value (pixel) from the frame from current experiemnt (right-lower corner of Arena A)
Bx = 482  # get this value (pixel) from the frame from current experiemnt (left-lower corner of Arena B)
y = 345  # get this value (pixel) from the frame from current experiemnt (any point on the border between Arena A and B)
laser_test = 0

movment_start = 0.5 # cm/s
movment_stop = 0.1  # cm/s

# pathway to file 

root = tk.Tk()  # creat main window 
root.withdraw()

df1 = filedialog.askopenfilename()  # choose file to upload 
df1 = df1.replace("/", "//") # Pythonic thing....
save_to = df1.replace(".csv", "")
df = pd.read_csv(df1)  # Creat data frame 


# data pre-procesing
df_drop = df.drop(range(0, starting_frame), axis=0)  # drop first frames -> due to rats puting into the test arena 
df_drop = df_drop.reset_index(drop=True)  # reste index 

df_drop["Item1_cm"] = df_drop["Item1"] *pix_to_cm 
df_drop["Item2_cm"] = df_drop["Item2"] *pix_to_cm 
df_drop["Item4_cm"] = df_drop["Item4"] *pix_to_cm 
df_drop["Item5_cm"] = df_drop["Item5"] *pix_to_cm 



df_drop.loc[df_drop["Item3"] < probability_treshold, ["Item1_cm", "Item2_cm"]] = float("NaN")   # set all values ( X and Y for head ) which were identified with probability below treshold  
df_drop.loc[df_drop["Item6"] < probability_treshold, ["Item4_cm", "Item5_cm"]] = float("NaN")   # set all values ( X and Y for body ) which were identified with probability below treshold  

df_drop.loc[df_drop["Item3"] < probability_treshold, ["Item1", "Item2"]] = float("NaN")   # set all values ( X and Y for head ) which were identified with probability below treshold  
df_drop.loc[df_drop["Item6"] < probability_treshold, ["Item4", "Item5"]] = float("NaN")   # set all values ( X and Y for body ) which were identified with probability below treshold  

print("Before interpolation:")
print(df_drop.isnull().sum())
df_drop = df_drop.interpolate()
print("After interpolation:")
print(df_drop.isnull().sum())





df_drop["Item1_cm"] = savgol_filter(df_drop["Item1_cm"], 21, 2) # smoothing data (x value of head) 
df_drop["Item2_cm"] = savgol_filter(df_drop["Item2_cm"], 21, 2) # smoothing data (y value of head) 
df_drop["Item4_cm"] = savgol_filter(df_drop["Item4_cm"], 21, 2) # smoothing data (x value of body)
df_drop["Item5_cm"] = savgol_filter(df_drop["Item5_cm"], 21, 2) # smoothing data (y value of body)

df_drop["Item1"] = savgol_filter(df_drop["Item1"], 21, 2) # smoothing data (x value of head) 
df_drop["Item2"] = savgol_filter(df_drop["Item2"], 21, 2) # smoothing data (y value of head) 
df_drop["Item4"] = savgol_filter(df_drop["Item4"], 21, 2) # smoothing data (x value of body)
df_drop["Item5"] = savgol_filter(df_drop["Item5"], 21, 2) # smoothing data (y value of body)
df_drop = df_drop.interpolate()    # interpolate all NaN value 

head = euclides(df_drop["Item1"], df_drop["Item2"], column="head_px")  # call for euclides function and calculate euclides distance for head
body = euclides(df_drop["Item4"], df_drop["Item5"], column="body_px")  # call for euclides function and calculate euclides distance for body 
head_cm = euclides(df_drop["Item1_cm"], df_drop["Item2_cm"], column="head_cm")  # call for euclides function and calculate euclides distance for head
body_cm = euclides(df_drop["Item4_cm"], df_drop["Item5_cm"], column="body_cm")  # call for euclides function and calculate euclides distance for body 


df_drop1 = pd.concat([df_drop, head, body, head_cm, body_cm], axis=1)   #  create finall data frame  

df_drop1["head_cm"] = savgol_filter(df_drop1["head_cm"], 21, 2)
df_drop1["body_cm"] = savgol_filter(df_drop1["body_cm"], 21, 2)

# data frame where data will be saved 
results = pd.DataFrame(np.nan, index = np.arange(1000), columns = ["area", "enter_frame", "exit_frame", "duration"])  # create new DF with define column name
results = results.fillna(0)  # fill all nan value with 0


#  wykryj szczura w pierwszej klatce
f = 0    # aktulany frejm
stop_me = True  # hamulec 
while stop_me == True:
    if ((df_drop1["Item1"][f] < Ax) & (df_drop1["Item2"][f] < y)) & ((df_drop1["Item4"][f] < Ax) & (df_drop1["Item5"][f] < y)):  #  check if rat is in area A (based on Ax and Y value)
        results["area"][0] = 'A'  # If rat is in arena A set column Arena first row to A
        results["enter_frame"][0] = f + 1  
        stop_me = False
    elif ((df_drop1["Item1"][f] > Bx) & (df_drop1["Item2"][f] < y)) & ((df_drop1["Item4"][f] > Bx) & (df_drop1["Item5"][f] < y)): #  check if rat is in area B (based on Bx and Y value)
        results["area"][0] = 'B'  
        results["enter_frame"][0] = f + 1
        stop_me = False
    elif ((Ax < df_drop1["Item1"][f] < Bx) & (df_drop1["Item2"][f] > y)) & ((Ax < df_drop1["Item4"][0] < Bx) & (df_drop1["Item5"][0] > y)):  #  check if rat is in area C (based on Ax,  Bx and Y value)
        results["area"][0] = 'C'
        results["enter_frame"][0] =  f + 1
        stop_me = False
    else:
        f += 1  # If rat is a superposition of A/B or A/C arenas take next frame 
# kolejne klatki
k = 0   # wiersze
for i in range(f + 1, len(df_drop1)):  # lecimy sobie przez wszystkie klatki
    if results["area"][k] == 'A':  # jesli  szczur jest w arenia A to wchodzi tutaj  
        if ((df_drop1["Item1"][i] > Ax) & (df_drop1["Item2"][i] > y)) & ((df_drop1["Item4"][i] > Ax) & (df_drop1["Item5"][i] > y)): #sprawdza czy szczur jest dalej w A, jesli jest to nie wchodzi i iterator sie zwieksza. JESLI SZCZUR JEST POZA A CZYLI W C TO WCHODZI
            results["exit_frame"][k] = i-1 # exit frame ustawia doputy szczur tam siedzial 
            results["duration"][k] = results["exit_frame"][k] - results["enter_frame"][k]  # duration
            k += 1  # przesuuwamy się o wiersz niżej
            results["area"][k] = 'C'  # stawiamy nowa arene w ktorej znjaduje sie szczur 
            results["enter_frame"][k] = i # klatka od ktorej se siedzi ziomek w arenie C 
    elif results["area"][k] == 'B': # jesli  szczur jest w arenia B to wchodzi tutaj  
        if ((df_drop1["Item1"][i] < Bx) & (df_drop1["Item2"][i] > y)) & ((df_drop1["Item4"][i] < Bx) & (df_drop1["Item5"][i] > y)): #sprawdza czy szczur jest dalej w B, jesli jest to nie wchodzi i iterator sie zwieksza. JESLI SZCZUR JEST POZA A CZYLI W C TO WCHODZI
            results["exit_frame"][k] = i-1 # exit frame ustawia doputy szczur tam siedzial 
            results["duration"][k] = results["exit_frame"][k] - results["enter_frame"][k]  # duration
            k += 1  # przesuuwamy się o wiersz niżej
            results["area"][k] = 'C'  # stawiamy nowa arene w ktorej znjaduje sie szczur 
            results["enter_frame"][k] = i # klatka od ktorej se siedzi ziomek w arenie C
    elif results["area"][k] == 'C': 
        if ((df_drop1["Item1"][i] < Ax) & (df_drop1["Item2"][i] < y)) & ((df_drop1["Item4"][i] < Ax) & (df_drop1["Item5"][i] < y)):  # sprawdza czy szczur jest w "A"
            results["exit_frame"][k] = i-1
            results["duration"][k] = results["exit_frame"][k] - results["enter_frame"][k]  # duration
            k += 1  # przesuuwamy się o wiersz niżej
            results["area"][k] = 'A'  # stawiamy nowa arene w ktorej znjaduje sie szczur 
            results["enter_frame"][k] = i # klatka od ktorej se siedzi ziomek w arenie A
        elif ((df_drop1["Item1"][i] > Bx) & (df_drop1["Item2"][i] < y)) & ((df_drop1["Item4"][i] > Bx) & (df_drop1["Item5"][i] < y)):  # sprawdza czy szczur jest w "B"
            results["exit_frame"][k] = i-1
            results["duration"][k] = results["exit_frame"][k] - results["enter_frame"][k]  # duration
            k += 1  # przesuuwamy się o wiersz niżej
            results["area"][k] = 'B'  # stawiamy nowa arene w ktorej znjaduje sie szczur 
            results["enter_frame"][k] = i # klatka od ktorej se siedzi ziomek w arenie B
# Ostatnia klatka 
for i in range(len(results)):  
    if results["exit_frame"][i] == 0: # find first 0 value 
        results["exit_frame"][i] = len(df_drop1) - (f+1)  # calculate last exit frame 
        results["duration"][i] =  results["exit_frame"][i] - results["enter_frame"][i] # calculate duration
        break

# analysis of movement episodes

state = -1 #unknow
list_of_states = []
list_of_states.append(state)
for i in df_drop1["body_cm"][1:]:
    if i > movment_start:
        state = 1
    elif i < movment_stop:
        state = 0
    list_of_states.append(state)

df_drop1["state_move"] = list_of_states

epi_total_mobile = 0
epi_total_immobile = 0
controller_total_mobile = True
controller_total_immobile = True

epi_a_mobile = 0
epi_a_immobile = 0
controller_a_mobile = True
controller_a_immobile = True

epi_b_mobile = 0
epi_b_immobile = 0
controller_b_mobile = True
controller_b_immobile = True

for i,j in enumerate(df_drop1["state_move"]):
    if j == 1:
        if controller_total_mobile:
            epi_total_mobile +=1
            controller_total_mobile = False
            controller_total_immobile = True
    elif j == 0:
        if controller_total_immobile:
            epi_total_immobile +=1
            controller_total_mobile = True
            controller_total_immobile = False
    
    if (df_drop1.loc[i, "Item1"] < Ax) & (df_drop1.loc[i, "Item2"] < y) & (df_drop1.loc[i, "Item4"] < Ax) & (df_drop1.loc[i, "Item5"] < y):
        if j == 1:
            if controller_a_mobile:
                epi_a_mobile += 1
                controller_a_mobile = False
                controller_a_immobile = True
        elif j == 0:
            if controller_a_immobile:
                epi_a_immobile += 1
                controller_a_mobile = True
                controller_a_immobile = False
    elif (df_drop1.loc[i, "Item1"] > Bx) & (df_drop1.loc[i, "Item2"] < y) & (df_drop1.loc[i, "Item4"] > Bx) & (df_drop1.loc[i, "Item5"] < y):
        if j == 1:
            if controller_b_mobile:
                epi_b_mobile += 1
                controller_b_mobile = False
                controller_b_immobile = True
        elif j == 0:
            if controller_b_immobile:
                epi_b_immobile += 1
                controller_b_mobile = True
                controller_b_immobile = False
    else:
        controller_a_mobile = True
        controller_a_immobile = True
        controller_b_mobile = True
        controller_b_immobile = True
        
# anlysis of movement speed
total_time = round(len(df_drop1)/30)
total_speed_imm_mmo = round(sum(df_drop1["body_cm"])/total_time,2)
total_speed_mmo = round(sum(df_drop1.loc[(df_drop1["state_move"] == 1), "body_cm"]) / (len(df_drop1.loc[df_drop1["state_move"] == 1, "state_move"])/30),2)
a_speed_mmo = round(sum(df_drop1.loc[(df_drop1["state_move"] == 1) & (df_drop1["Item1"] < Ax) & (df_drop1["Item2"] < y) & (df_drop1["Item4"] < Ax) & (df_drop1["Item5"] < y), "body_cm"]) / (len((df_drop1.loc[(df_drop1["state_move"] == 1) & (df_drop1["Item1"] < Ax) & (df_drop1["Item2"] < y) & (df_drop1["Item4"] < Ax) & (df_drop1["Item5"] < y), "state_move"]))/30),2)
b_speed_mmo = round(sum(df_drop1.loc[(df_drop1["state_move"] == 1) & (df_drop1["Item1"] > Bx) & (df_drop1["Item2"] < y) & (df_drop1["Item4"] > Bx) & (df_drop1["Item5"] < y), "body_cm"]) / (len((df_drop1.loc[(df_drop1["state_move"] == 1) & (df_drop1["Item1"] > Bx) & (df_drop1["Item2"] < y) & (df_drop1["Item4"] > Bx) & (df_drop1["Item5"] < y), "state_move"]))/30),2)



duration_cpp = round(sum(results["duration"])/30)  # calculate duration of test
duration_in_a = round(sum(results.loc[results["area"] == "A", "duration"])/30)  # calculate duration of  rat being in A arena 
percent_A = round((duration_in_a/duration_cpp * 100), 2)
duration_in_b = round(sum(results.loc[results["area"] == "B", "duration"])/30)    # calculate duration of  rat being in B arena 
percent_B = round((duration_in_b/duration_cpp * 100), 2)

duration_in_c = round(sum(results.loc[results["area"] == "C", "duration"])/30)   # calculate duration of  rat being in A arena 
percent_C = round((duration_in_c/duration_cpp * 100), 2)

list_area = list(results["area"])
entrance_a = list_area.count("A") # calculate entrance to A
entrance_b = list_area.count("B")  # calculate entrance to B

mean_time_a = round(duration_in_a / entrance_a)   #  calculate Mean time of rat being in A [s]
mean_time_b = round(duration_in_b / entrance_b)  #  calculate Mean time of rat being in B [s]
min_time_in_a = round(results.loc[results["area"] == "A", "duration"].min()/30)
min_time_in_b = round(results.loc[results["area"] == "B", "duration"].min()/30)
max_time_in_a = round(results.loc[results["area"] == "A", "duration"].max()/30)
max_time_in_b = round(results.loc[results["area"] == "B", "duration"].max()/30)
median_time_in_a = round(results.loc[results["area"] == "A", "duration"].median()/30)
median_time_in_b = round(results.loc[results["area"] == "B", "duration"].median()/30)
total_distance = round((df_drop1["body_cm"][0:len(df_drop1["body_cm"])].sum(skipna = True))) # calculate distance traveled
total_distance_a = float(round((df_drop1.loc[((df_drop1["Item1"] < Ax) & (df_drop1["Item2"] < y)) & ((df_drop1["Item4"] < Ax) & (df_drop1["Item5"] < y)), ["body_cm"]].sum(skipna = True)),2))  # calculate distance traveled in Arena A 
total_distance_b = float(round((df_drop1.loc[((df_drop1["Item1"] > Bx) & (df_drop1["Item2"] < y)) & ((df_drop1["Item4"] > Bx) & (df_drop1["Item5"] < y)), ["body_cm"]].sum(skipna = True)),2))  # calculate distance traveled in Arena B 

finall_results = pd.DataFrame(columns = ["Time in CPP", "Time in A [s]", "Time in A [%]", "Mean time in A [s]", "Min time in A [s]", "Max time in A [s]", "Median time in A [s]", "Time in B [s]", "Time in B [%]", "Mean time in B [s]", "Min time in B [s]", "Max time in B [s]", "Median time in B [s]","Time in C [s]", "Time in C [%]", "Entrance to A", "Entrance to B", "Stim_at", "Total distance [cm]", "Total distance in A[cm]", "Total distance in B[cm]", "Episode of total mobile", "Episode of total immobile", "Episode of mobile in a", "Episode of immobile in a", "Episode of mobile in b", "Episode of immobile in b", "Total velocity", "Total velocity in mob", "Mob velocity in a", "Mob velocity in b"] ) # final data frame


# where is stim? more WHEN IS STIM :0
stop_me1 = True
    
while stop_me1 == True: # in general: find first frame with rat in Arena A or  B check if laser item is True then set where stim was 
    if ((df_drop1["Item1"][laser_test] < Ax) & (df_drop1["Item2"][laser_test] < y)) & ((df_drop1["Item4"][laser_test] < Ax) & (df_drop1["Item5"][laser_test] < y)):
        if  df_drop1["Item7"][laser_test] == True:
            stim_at = "A"
            stop_me1 = False
        else:
            stim_at = "B"
            stop_me1 = False
    elif ((df_drop1["Item1"][laser_test] > Bx) & (df_drop1["Item2"][laser_test] < y)) & ((df_drop1["Item4"][laser_test] > Bx) & (df_drop1["Item5"][laser_test] < y)):
        if  df_drop1["Item7"][laser_test] == True:
            stim_at =  "B"
            stop_me1 = False
        else:
            stim_at = "A"
            stop_me1 = False
    elif ((Ax < df_drop1["Item1"][laser_test] < Bx) & (df_drop1["Item2"][laser_test] > y)) & ((Ax < df_drop1["Item4"][laser_test] < Bx) & (df_drop1["Item5"][laser_test] > y)):
        laser_test += 1 
    else:
        laser_test += 1
x =[total_time, duration_in_a, percent_A, mean_time_a, min_time_in_a, max_time_in_a, median_time_in_a, duration_in_b, percent_B, mean_time_b, min_time_in_b, max_time_in_b, median_time_in_b, duration_in_c, percent_C, entrance_a, entrance_b, stim_at, total_distance, total_distance_a, total_distance_b, epi_total_mobile, epi_total_immobile, epi_a_mobile, epi_a_immobile, epi_b_mobile, epi_b_immobile, total_speed_imm_mmo, total_speed_mmo, a_speed_mmo, b_speed_mmo]
finall_results.loc[0, :] = [total_time, duration_in_a, percent_A, mean_time_a, min_time_in_a, max_time_in_a, median_time_in_a, duration_in_b, percent_B, mean_time_b, min_time_in_b, max_time_in_b, median_time_in_b, duration_in_c, percent_C, entrance_a, entrance_b, stim_at, total_distance, total_distance_a, total_distance_b, epi_total_mobile, epi_total_immobile, epi_a_mobile, epi_a_immobile, epi_b_mobile, epi_b_immobile, total_speed_imm_mmo, total_speed_mmo, a_speed_mmo, b_speed_mmo]
finall_results.to_excel(save_to + "\\" + "result" + ".xlsx")

# Plot rats travelled distance 
# =============================================================================
# fig, ax = plt.subplots(figsize=(14, 6))    # creat fig object and ax object
# fig.suptitle("CPP/A tracing", fontsize=16) 
# ax.plot(df_drop1["Item1"], df_drop1["Item2"] * -1, "-r", alpha = 0.8) # plot rat walk pathway
# ax.set_aspect('equal')
# ax.set_xlabel("X position in pixel")  
# ax.set_ylabel("Y position in pixel")
# left_arena = plt.legend([f'Time in arena A: {finall_results["Time in A [%]"][0]}%'], loc = 3)  
# right_arena = plt.legend([f'Time in arena B: {finall_results["Time in B [%]"][0]}%'], loc = 4)
# ax.add_artist(left_arena)
# ax.add_artist(right_arena)
# ax.xaxis.set_ticks([])
# ax.yaxis.set_ticks([])
# #fig.tight_layout(left_arena)
# plt.show()
# rat_id_group = input("Enter rat id + group join by underscroe (ex. zw1_ctrl): ")
# #sf =  f"F:\\Behawior\\Kohorta_D\\2021_06_13_CPPA_conditioning_day1\\{rat_id_group}\\{rat_id_group}_plot.png"  # save fig to
# sf = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
# sf = sf + f"\\{rat_id_group}_plot.svg"
# fig.savefig(sf)
# # Plot rats travelled KDE
# density_plot = sns.kdeplot(df_drop1["Item1"], df_drop1["Item2"] * -1, shade=True, shade_lowest=False, bw = 0.15, cmap = "rainbow", cbar = True)  # plot kerenel density estimation of rat travelled pathway
# density_plot.set_aspect('equal')
# density_plot.set_facecolor("midnightblue")
# density_plot.yaxis.set_ticks([])
# density_plot.set_xlabel("X position in pixel")
# density_plot.set_ylabel("Y position in pixel")
# density_plot.set_title("CPP/A tracing density map")
# fig2 = density_plot.get_figure()
# #sf2 = f"F:\\Behawior\\Kohorta_D\\2021_06_13_CPPA_conditioning_day1\\{rat_id_group}\\{rat_id_group}_density_plot.png" # save fig to
# sf2 = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")
# sf2 = sf2 + f"\\{rat_id_group}_density_plot.svg"
# fig2.savefig(sf2)
# #cbar = density_plot.colorbar
# =============================================================================
