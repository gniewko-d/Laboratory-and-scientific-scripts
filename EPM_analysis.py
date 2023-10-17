import easygui
import pandas as pd
import numpy as np
from termcolor import colored
import cv2
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import tkinter as tk
import seaborn as sns;# sns.set_theme()
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import math
from statistics import mean
import os
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


# VARIABLES
df = None
EPM_cmm = 14 # lenght of center zone diameter
probability_treshold = 0.9 
test_data = None
movment_start = 0.5
movment_stop = 0.1

class EPM_object:
    def __init__(self, opto = False):
        self.opto = opto
        self.open_video()
    
    def open_video(self):
        self.points_description_list = [ "up_corresponding_open", "up_corresponding_close", "center zone right point", "right_corresponding_open", "right_corresponding_close", "center zone down point", "down_corresponding_open", "down_corresponding_close", "center zone left point", "left_corresponding_open", "left_corresponding_close", "midpoint on the outer boundary of the north-east open arm", "midpoint on the outer boundary of the south-west open arm"]
        self.points_description = (i for i in self.points_description_list)
        self.boundry_points = []
        easygui.msgbox('Choose EMP video [.mp4, .AVI]')
        self.video_file = easygui.fileopenbox(title="Select a EMP video file", filetypes= ["*.mp4", "*AVI"],  multiple=False)
        easygui.msgbox('Choose EMP CSV files')
        self.list_of_files = easygui.fileopenbox(title="Select a CSV file of EMP", filetypes= "*.csv",  multiple=True)
        self.rat_id = []
        for i in self.list_of_files:
            first = i.split("\\")
            second = first[-1].split("_")[0:3]
            finall = "_".join(second)
            self.rat_id.append(finall)
        self.data_n = int(len(self.list_of_files))
        self.cap = cv2.VideoCapture(self.video_file)
        #self.cap = cv2.VideoCapture("C:/Users/gniew/Desktop/Data_EPM/zw7_exp_kohAprimDLC_resnet50_EMP_V1Feb25shuffle1_450000_filtered_labeled.mp4")
        amount_of_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_number = random.randint(1, amount_of_frames)
        cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, self.frames = self.cap.read()
        # displaying the image
       
        cv2.setMouseCallback("screen", self.click_event)
        cv2.imshow("screen", self.frames)
        easygui.msgbox("click  with the left mouse on: center zone up point within EMP")
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
        self.draw_lines()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"fps: {self.fps}")
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
    # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
            self.boundry_points.append((x,y))
            try:
                current_point = next(self.points_description)
                easygui.msgbox("click on: " + current_point)
                
            except StopIteration:
                cv2.destroyAllWindows()
                easygui.msgbox("All points are obtained")
                self.points_description_list.insert(0, "center zone up point")
                self.points_disct = dict(zip(self.points_description_list, self.boundry_points))

    def draw_lines(self):
        global EPM_cmm
        NE_center_zone = [self.points_disct["center zone up point"],self.points_disct["center zone right point"]]
        SE_center_zone = [self.points_disct["center zone down point"],self.points_disct["center zone right point"]]
        SW_center_zone = [self.points_disct["center zone down point"],self.points_disct["center zone left point"]]
        NW_center_zone = [self.points_disct["center zone up point"],self.points_disct["center zone left point"]]
        NE_open_arm_up = [self.points_disct["center zone up point"],self.points_disct["up_corresponding_open"]]
        NE_open_arm_down = [self.points_disct["center zone right point"],self.points_disct["right_corresponding_open"]]
        SW_open_arm_up = [self.points_disct["center zone left point"],self.points_disct["left_corresponding_open"]]
        SW_open_arm_down = [self.points_disct["center zone down point"],self.points_disct["down_corresponding_open"]]
        NW_close_arm_up = [self.points_disct["center zone up point"],self.points_disct["up_corresponding_close"]]
        NW_close_arm_down = [self.points_disct["center zone left point"],self.points_disct["left_corresponding_close"]]
        SE__close_arm_up = [self.points_disct["center zone right point"],self.points_disct["right_corresponding_close"]]
        SE__close_arm_down = [self.points_disct["center zone down point"],self.points_disct["down_corresponding_close"]]
        NE_outter_zone = [self.points_disct["up_corresponding_open"],self.points_disct["right_corresponding_open"]]
        SE_outter_zone = [self.points_disct["down_corresponding_close"],self.points_disct["right_corresponding_close"]]
        SW_outter_zone = [self.points_disct["down_corresponding_open"],self.points_disct["left_corresponding_open"]]
        NW_outter_zone = [self.points_disct["up_corresponding_close"],self.points_disct["left_corresponding_close"]]
        self.poly_NE = Polygon([self.points_disct["center zone up point"], self.points_disct["up_corresponding_open"], self.points_disct["right_corresponding_open"], self.points_disct["center zone right point"]])
        self.poly_SE = Polygon([self.points_disct["center zone right point"], self.points_disct["right_corresponding_close"], self.points_disct["down_corresponding_close"], self.points_disct["center zone down point"]])
        self.poly_SW = Polygon([self.points_disct["center zone left point"], self.points_disct["center zone down point"], self.points_disct["down_corresponding_open"], self.points_disct["left_corresponding_open"]])
        self.poly_NW = Polygon([self.points_disct["center zone left point"], self.points_disct["left_corresponding_close"], self.points_disct["up_corresponding_close"], self.points_disct["center zone up point"]])
        self.poly_center = Polygon([self.points_disct["center zone up point"], self.points_disct["center zone right point"], self.points_disct["center zone down point"], self.points_disct["center zone left point"]])
        self.boundry_container = [NE_center_zone, SE_center_zone, SW_center_zone, NW_center_zone, NE_open_arm_up, NE_open_arm_down, SW_open_arm_up, SW_open_arm_down, NW_close_arm_up, NW_close_arm_down, SE__close_arm_up, SE__close_arm_down, NE_outter_zone, SE_outter_zone, SW_outter_zone, NW_outter_zone]
        self.poly_container = [self.poly_NE, self.poly_SE, self.poly_SW, self.poly_NW, self.poly_center]
        self.outer_NE = self.points_disct["midpoint on the outer boundary of the north-east open arm"]
        self.outer_SW = self.points_disct["midpoint on the outer boundary of the south-west open arm"]
        
        EPM_pix = abs(self.points_disct["center zone up point"][1] - self.points_disct["center zone down point"][1])
        self.pix_to_cm = EPM_cmm/ EPM_pix
        
        self.outer_NE_cm = [i * self.pix_to_cm for i in self.outer_NE]
        self.outer_SW_cm = [i * self.pix_to_cm for i in self.outer_SW]
        
        for i in self.boundry_container:
            x = [i[0][0], i[1][0]]
            y = [i[0][1], i[1][1]]
            plt.plot(x, y, color="white", linewidth=2)
        plt.imshow(self.frames)
        plt.show()
       
    def polygen_check(self, head, body):
       
        if self.poly_center.contains(head) and self.poly_center.contains(body):
                self.rows_init[0], self.rows_init[1] = 1, 1
                if self.current_poly_v1 == "SE_CLOSE" or self.current_poly_v1 == "NW_CLOSE":
                    self.entry_center += 1
                if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
                
                self.current_poly = "center"
                self.current_poly_v1 = "center"
                
        elif self.poly_NE.contains(head) and self.poly_center.contains(body):
                self.rows_init[1], self.rows_init[2] = 1, 1
                if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
                
                
                self.current_poly = "center_body_NE_OPEN_head"
                
                    
        elif self.poly_NE.contains(body) and self.poly_center.contains(head):
            self.rows_init[0], self.rows_init[3] = 1, 1
            self.current_poly = "center_head_NE_OPEN_body"
            
                
        elif self.poly_SE.contains(head) and self.poly_center.contains(body):
            self.rows_init[1], self.rows_init[4] = 1, 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            
            self.current_poly = "center_body_SE_CLOSE_head"
                
        elif self.poly_SE.contains(body) and self.poly_center.contains(head):
            self.rows_init[0], self.rows_init[5] = 1, 1
            if self.current_poly == "SE_CLOSE":
                self.e_h_c_b_c_start = True
                self.e_h_c_b_c_start_v2 = True
            
            
            self.current_poly = "center_head_SE_CLOSE_body"
                
        elif self.poly_SW.contains(head) and self.poly_center.contains(body):
            self.rows_init[1], self.rows_init[6] = 1, 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            
            self.current_poly = "center_body_SW_OPEN_head"
                
        elif self.poly_SW.contains(body) and self.poly_center.contains(head):
            self.rows_init[0], self.rows_init[7] = 1, 1
            self.current_poly = "center_head_SW_OPEN_body"
                
        elif self.poly_NW.contains(head) and self.poly_center.contains(body):
            self.rows_init[1], self.rows_init[8] = 1, 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
                      
            self.current_poly = "center_body_NW_CLOSE_head"
                
        elif self.poly_NW.contains(body) and self.poly_center.contains(head):
            self.rows_init[0], self.rows_init[9] = 1, 1
            if self.current_poly == "NW_CLOSE":
                self.e_h_c_b_c_start = True
                self.e_h_c_b_c_start_v2 =  True
            self.current_poly = "center_head_NW_CLOSE_body"
            
        elif self.poly_NE.contains(head) and self.poly_NE.contains(body):
            self.rows_init[2], self.rows_init[3] = 1, 1
            if self.current_poly_v1 != "NE_OPEN" :
                self.entry_open += 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            self.current_poly = "NE_OPEN"
            self.current_poly_v1 = "NE_OPEN"

        elif self.poly_SE.contains(head) and self.poly_SE.contains(body):
            self.rows_init[4], self.rows_init[5] = 1, 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            
            self.current_poly = "SE_CLOSE"
            self.current_poly_v1 = "SE_CLOSE"
            self.entry_close = True

        elif self.poly_SW.contains(head) and self.poly_SW.contains(body):
            self.rows_init[6], self.rows_init[7] = 1, 1
            if self.current_poly_v1 != "SW_OPEN" :
                self.entry_open += 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            self.current_poly = "SW_OPEN"
            self.current_poly_v1 = "SW_OPEN"
                
        elif self.poly_NW.contains(head) and self.poly_NW.contains(body):
            self.rows_init[8], self.rows_init[9] = 1, 1
            if (self.current_poly == "center_head_SE_CLOSE_body" or self.current_poly == "center_head_NW_CLOSE_body") and self.e_h_c_b_c_start_v2:
                    self.e_h_c_b_c_stop = True
            
            self.entry_close = True
            self.current_poly = "NW_CLOSE"
            self.current_poly_v1 = "NW_CLOSE"
                
        elif (self.poly_NE.contains(head) == False) and self.poly_NE.contains(body):
            self.rows_init[10], self.rows_init[3] = 1, 1
            self.current_poly = "head_out_NE_OPEN_body"
                
        elif self.poly_NE.contains(head) and (self.poly_NE.contains(body) == False):
            self.rows_init[2], self.rows_init[11] = 1, 1
            self.current_poly = "body_out_NE_OPEN_head"
                
        elif (self.poly_SE.contains(head) == False) and self.poly_SE.contains(body):
            self.rows_init[10], self.rows_init[5] = 1, 1
            self.current_poly = "head_out_SE_CLOSE_body"
                
        elif self.poly_SE.contains(head) and (self.poly_SE.contains(body) == False):
            self.rows_init[4], self.rows_init[11] = 1, 1
            self.current_poly = "body_out_SE_CLOSE_head"

        elif (self.poly_SW.contains(head) == False) and self.poly_SW.contains(body):
            self.rows_init[10], self.rows_init[7] = 1, 1
            self.current_poly = "head_out_SW_OPEN_body"
                
        elif self.poly_SW.contains(head) and (self.poly_SW.contains(body)== False):
            self.rows_init[6], self.rows_init[11] = 1, 1
            self.current_poly = "body_out_SW_OPEN_head"

        elif (self.poly_NW.contains(head) == False) and self.poly_NW.contains(body):
            self.rows_init[10], self.rows_init[9] = 1, 1
            self.current_poly = "head_out_NW_CLOSE_body"

        elif self.poly_NW.contains(head) and (self.poly_NW.contains(body)== False):
            self.rows_init[8], self.rows_init[11] = 1, 1
            self.current_poly = "body_out_NW_CLOSE_head"
                
        elif (self.poly_center.contains(head)== False) and self.poly_center.contains(body):
            self.rows_init[10], self.rows_init[1] = 1, 1
            self.current_poly = "head_out_center_body"
                    
        elif self.poly_center.contains(head) and (self.poly_center.contains(body) == False):
            self.rows_init[0], self.rows_init[11] = 1, 1
            self.current_poly = "body_out_center_head"
        
        self.rows_init.append(self.current_poly)
    
    def pre_processing(self, mode = "normal"):
        global EPM_cmm, probability_treshold 
        self.gen_df = (pd.read_csv(i, header=None) for i in self.list_of_files) 
        if mode == "normal":
            self.gen_df_v2 = []
            print("pre_porcessing  begine")
            columns_name_opto = ["Nose_x", "Nose_y", "Nose_prob", "Head_x", "Head_y", "Head_prob", "LeftEar_x",	"LeftEar_y", "LeftEar_prob", "RightEar_x", "RightEar_y", "RightEar_prob", "BodyCntr_x",	"BodyCntr_y", "BodyCntr_prob", "TailBase_x", "TailBase_y", "TailBase_prob", "Opto_x", "Opto_y", "Opto_prob"]
            columns_name = ["Nose_x", "Nose_y", "Nose_prob", "Head_x", "Head_y", "Head_prob", "LeftEar_x",	"LeftEar_y", "LeftEar_prob", "RightEar_x", "RightEar_y", "RightEar_prob", "BodyCntr_x",	"BodyCntr_y", "BodyCntr_prob", "TailBase_x", "TailBase_y", "TailBase_prob"]
            for i,j in enumerate(self.gen_df):
                j = j.iloc[3:,1:].reset_index(drop = True)
                if self.opto:
                    j.columns = columns_name_opto
                    j = j.loc[:, ["Head_x", "Head_y", "Head_prob", "BodyCntr_x", "BodyCntr_y",  "BodyCntr_prob", "Old_index", "Opto_x", "Opto_y", "Opto_prob"]]
                else:
                    j.columns = columns_name
                    j = j.loc[:, ["Head_x", "Head_y", "Head_prob", "BodyCntr_x", "BodyCntr_y",  "BodyCntr_prob", "Old_index"]]
                self.gen_df_v2.append(j)
        elif mode == "GD_666":
             columns_name = ["Nose_x", "Nose_y", "Nose_prob", "Head_x", "Head_y", "Head_prob", "LeftEar_x",	"LeftEar_y", "LeftEar_prob", "RightEar_x", "RightEar_y", "RightEar_prob", "BodyCntr_x",	"BodyCntr_y", "BodyCntr_prob", "TailBase_x", "TailBase_y", "TailBase_prob", "Opto_x", "Opto_y", "Opto_prob", "Old_index"]
             self.gen_df_v2 = []
             self.drop_configuration = ["C:/Users/gniew/Desktop/Data_EPM/drop_list_koh_F.xlsx"]
             print("!WELCOME!")
             print("loaded configuration")
             ans = pd.read_excel(self.drop_configuration[0])
             ans = [ans.iloc[i,:].values.flatten().tolist() for i in range(len(ans))] 
             ans = [i[1:] for i in ans]
             drop_list = [i[0:10] for i in ans]
             opto_list_start = [i[10] for i in ans]
             opto_list_stop = [i[11] for i in ans]
             self.opto_duration = [[a, b] for a, b in zip(opto_list_start, opto_list_stop)]

             for i,j in enumerate(self.gen_df):
                 x = [str(xd) for xd in drop_list[i]]
                 str_lis = [int(float(xdd)) for xdd in x if xdd != "nan"]
                 j = j.iloc[3:,1:].reset_index(drop = True)
                 for k in range(0, len(str_lis), 2):
                    j.iloc[str_lis[k]:str_lis[k+1]] = np.nan
                 j.dropna(inplace=True)
                 index = j.index 
                 j["old_index"] = index
                 j.reset_index(inplace = True, drop = True)
                 j.columns = columns_name
                 j = j.loc[:, ["Head_x", "Head_y", "Head_prob", "BodyCntr_x", "BodyCntr_y",  "BodyCntr_prob", "Old_index"]]
                 self.gen_df_v2.append(j)
        self.gen_df_v2 = (i for i in self.gen_df_v2)
        self.gen_df = []
        self.travelled_plot = []
        for index, data_frame in enumerate(self.gen_df_v2):
            data_frame = data_frame.apply(pd.to_numeric)
            data_frame["Head_x_cm"] = data_frame["Head_x"] * self.pix_to_cm
            data_frame["Head_y_cm"] = data_frame["Head_y"] * self.pix_to_cm
            data_frame["BodyCntr_x_cm"] = data_frame["BodyCntr_x"] * self.pix_to_cm
            data_frame["BodyCntr_y_cm"] = data_frame["BodyCntr_y"] * self.pix_to_cm

            data_frame.loc[data_frame["Head_prob"] < probability_treshold, ["Head_x_cm", "Head_y_cm"]] = float("NaN")   # set all values ( X and Y for head ) which were identified with probability below treshold  
            data_frame.loc[data_frame["BodyCntr_prob"] < probability_treshold, ["BodyCntr_x_cm", "BodyCntr_y_cm"]] = float("NaN")   

            data_frame.loc[data_frame["Head_prob"] < probability_treshold, ["Head_x", "Head_y"]] = float("NaN")   # set all values ( X and Y for head ) which were identified with probability below treshold  
            data_frame.loc[data_frame["BodyCntr_prob"] < probability_treshold, ["BodyCntr_x", "BodyCntr_y"]] = float("NaN") 

            print("Before interpolation:")
            print(data_frame.isnull().sum())
            data_frame = data_frame.interpolate()
            print("After interpolation:")
            print(data_frame.isnull().sum())

            data_frame["Head_x_cm"] = savgol_filter(data_frame["Head_x_cm"], 20, 2) # smoothing data (x value of head) 
            data_frame["Head_y_cm"] = savgol_filter(data_frame["Head_y_cm"], 20, 2) # smoothing data (y value of head) 
            data_frame["BodyCntr_x_cm"] = savgol_filter(data_frame["BodyCntr_x_cm"], 20, 2) # smoothing data (x value of body)
            data_frame["BodyCntr_y_cm"] = savgol_filter(data_frame["BodyCntr_y_cm"], 20, 2) # smoothing data (y value of body)

            data_frame["Head_x"] = savgol_filter(data_frame["Head_x"], 20, 2) # smoothing data (x value of head) 
            data_frame["Head_y"] = savgol_filter(data_frame["Head_y"], 20, 2) # smoothing data (y value of head) 
            data_frame["BodyCntr_x"] = savgol_filter(data_frame["BodyCntr_x"], 20, 2) # smoothing data (x value of body)
            data_frame["BodyCntr_y"] = savgol_filter(data_frame["BodyCntr_y"], 20, 2) # smoothing data (y value of body)

            data_frame = data_frame.interpolate()   # interpolate all NaN value 

            head = self.euclides(data_frame["Head_x"], data_frame["Head_y"], column="head_cm")  # call for euclides function and calculate euclides distance for head
            body = self.euclides(data_frame["BodyCntr_x"], data_frame["BodyCntr_y"], column="body_px")  # call for euclides function and calculate euclides distance for body 
            head_cm = self.euclides(data_frame["Head_x_cm"], data_frame["Head_y_cm"], column="head_cm")  # call for euclides function and calculate euclides distance for head
            body_cm = self.euclides(data_frame["BodyCntr_x_cm"], data_frame["BodyCntr_y_cm"], column="body_cm")  # call for euclides function and calculate euclides distance for body 
            data_frame = pd.concat([data_frame, head, body, head_cm, body_cm], axis=1) 
            self.list_of_states = self.movment_state(data = data_frame)
            data_frame = pd.concat([data_frame,self.list_of_states], axis=1) 
            self.gen_df.append(data_frame)
            
            head_x_graph, head_y_graph = data_frame["Head_x"].tolist(), data_frame["Head_y"].tolist()

            self.travelled_plot.append([head_x_graph, head_y_graph])
        self.gen_df = (ii for ii in self.gen_df)
    
    def movment_state(self,data):
        global movment_start, movment_stop
        state = -1 #unknow
        list_of_states = []
        list_of_states.append(state)
        for i in data["body_cm"][1:]:
            if i > movment_start:
                state = 1
            elif i < movment_stop:
                state = 0
            list_of_states.append(state)
        series_of_states = pd.Series(list_of_states, name = "movment_state")
        return series_of_states

    def number_of_movement(self, data):
         epi_mobile = 0
         epi_immobile = 0
         controller_total_mobile = True
         controller_total_immobile = True
         for i,j in enumerate(data["movment_state"]):
            if j == 1:
                if controller_total_mobile:
                    epi_mobile +=1
                    controller_total_mobile = False
                    controller_total_immobile = True
            elif j == 0:
                if controller_total_immobile:
                    epi_immobile +=1
                    controller_total_mobile = True
                    controller_total_immobile = False
         return epi_mobile, epi_immobile

    def euclides(self,data1, data2, column: str): # creat function that will calculate euclides distance 
        ser = []  # creat empty list 
        for i in range(0, len(data1)-1):  #  iterate thorugh data witout last frame 
        
            point_1 = [data1[i], data2[i]]
            point_2 = [data1[i+1], data2[i+1]]
        
            e = math.dist(point_1, point_2)
            ser.append(e)       # append next distance 
        ser.insert(0,0) # first frame no distance moved 
        serx = pd.Series(ser, name = column)  # creat pd series 1
        return serx   #return pd series 

    def get_position(self, mode = "normal"):

        global test_data
        results = None
        self.results_container = []
        
        self.entry_open_list = []
        self.entry_open_list_bin1 = []
        self.entry_open_list_bin2 = []
        self.entry_open_list_bin3 = []
        
        self.entry_center_list = []
        self.entry_center_list_bin1 = []
        self.entry_center_list_bin2 = []
        self.entry_center_list_bin3 = []

        self.max_open_list = []
        self.max_open_list_bin1 = []
        self.max_open_list_bin2= []
        self.max_open_list_bin3 = []
        
        self.episodes_head_center_body_close_all = []
        self.e_h_c_b_c_start = False
        self.e_h_c_b_c_stop = False
        self.e_h_c_b_c_start_v2 = False
        
        self.backward_episodes_all = []
        
        for index, data_frame in enumerate(self.gen_df):
            self.current_poly = "Unknown"
            self.current_poly_v1 = "Unknown"
            self.duration = 0
            self.backward_episodes = []
            
            self.entry_close = False
            self.entry_open = 0
            self.entry_center = 0
            self.episodes_head_center_body_close = []

            if mode == "normal":
                index_list = data_frame.index.tolist()
                index_list_3 = np.array_split(index_list, 3)
                self.brodcasting_index_list_3 = index_list_3
            elif mode == "GD_666":
                 index_list_3 = []
                 index_list_3.append([0, min(self.opto_duration[index])])
                 index_list_3.append([min(self.opto_duration[index]), max(self.opto_duration[index])])
                 index_list_3.append([max(self.opto_duration[index]), len(data_frame)-4])
                 self.brodcasting_index_list_3 = index_list_3
                 
            results = pd.DataFrame(index = list(range(len(data_frame))), columns = ["center_head", "center_body", "NE_head", "NE_body", "SE_head", "SE_body", "SW_head", "SW_body", "NW_head", "NW_body", "head_out", "body_out", "Location"])
            
            self.previous_cm_NE = "unknown"
            self.previous_cm_NE_BODY = "unknown"
            self.punishment = 0 
            self.backward_type_1 = "stop"
            self.backward_type_2 = "stop"
            self.backward_controler = False

            min_current_cm = 200000000.0
            min_current_cm_bin = 200000000.0
      
            self.time_to_open_stoper = True
            self.time_to_close_stoper = True
            for i, rows in data_frame.iterrows():
                self.rows_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                head_point = data_frame.loc[i,["Head_x", "Head_y"]].tolist()
                head_point = Point(head_point[0],head_point[1])
                body_point =  data_frame.loc[i,["BodyCntr_x", "BodyCntr_y"]].tolist()
                body_point =  Point(body_point[0],body_point[1])
                
                head_point_cm = data_frame.loc[i,["Head_x_cm", "Head_y_cm"]].tolist()
                body_point_cm = data_frame.loc[i,["BodyCntr_x_cm", "BodyCntr_y_cm"]].tolist()
                
                self.current_cm_NE = math.dist(head_point_cm, self.outer_NE_cm)
                current_cm_SW = math.dist(head_point_cm, self.outer_SW_cm)
                self.current_cm_NE_BODY = math.dist(body_point_cm, self.outer_NE_cm)
                self.current_time = i +1

                if (self.current_cm_NE < min_current_cm_bin) and self.entry_close:
                     min_current_cm_bin = self.current_cm_NE
                     
                
                if (current_cm_SW < min_current_cm_bin) and self.entry_close:
                     min_current_cm_bin = current_cm_SW
                     

                if (self.current_cm_NE < min_current_cm) and self.entry_close:
                     min_current_cm = self.current_cm_NE
                     
                     

                if (current_cm_SW < min_current_cm) and self.entry_close: 
                     min_current_cm = current_cm_SW
                    
                
                self.polygen_check(head_point, body_point)
                
                
                if i == max(index_list_3[0]):
                    self.entry_open_list_bin1.append(self.entry_open)
                    self.entry_center_list_bin1.append(self.entry_center)
                    min_current_cm_bin 
                    ans1 = self.entry_open
                    ans1_1 = self.entry_center
                    self.max_open_list_bin1.append(round(min_current_cm_bin,2))
                    min_current_cm_bin =  200000000.0
                    

                elif i == max(index_list_3[1]):
                    ans2 = self.entry_open - ans1
                    ans2_2 = self.entry_center - ans1_1
                    self.entry_open_list_bin2.append(ans2)
                    self.entry_center_list_bin2.append(ans2_2)
                    self.max_open_list_bin2.append(round(min_current_cm_bin,2))
                    min_current_cm_bin =  200000000.0
                
                elif i == max(index_list_3[2]):
                    ans3 = self.entry_open - ans2 - ans1
                    ans3_3 = self.entry_center - ans2_2 - ans1_1
                    self.entry_open_list_bin3.append(ans3)
                    self.entry_center_list_bin3.append(ans3_3)
                    self.max_open_list_bin3.append(round(min_current_cm_bin,2))
                    
                results.iloc[i,:] =  self.rows_init
                if self.e_h_c_b_c_start:
                    start_1 = i
                    self.e_h_c_b_c_start = False
                    if i <  max(index_list_3[0]):
                         bin_stage = "bin_1"
                    elif i < max(index_list_3[1]):
                         bin_stage = "bin_2"
                    elif i < max(index_list_3[2]):
                         bin_stage = "bin_3"
                if self.e_h_c_b_c_stop:
                    stop_1 = i
                    self.e_h_c_b_c_stop = False
                    self.e_h_c_b_c_start_v2 =  False
                    duration = stop_1 - start_1
                    self.episodes_head_center_body_close.append([bin_stage, round(duration/self.fps,2)])
                self.backward()
                
                if self.time_to_open_stoper and self.entry_close and (self.current_poly == "NE_OPEN" or self.current_poly == "SW_OPEN"):
                     self.time_to_open = round((i+1)/self.fps,2)
                     self.time_to_open_stoper = False
                if self.time_to_close_stoper and self.entry_close and self.current_poly == "center":
                     self.time_to_close = round((i+1)/self.fps,2)
                     self.time_to_close_stoper = False

            self.min_current_cm = round(min_current_cm,2)
            self.max_open_list.append(self.min_current_cm)
            
            self.episodes_head_center_body_close_all.append(self.episodes_head_center_body_close)
            self.backward_episodes_all.append(self.backward_episodes)
            self.entry_open_list.append(self.entry_open)
            self.entry_center_list.append(self.entry_center)
            self.list_of_states = self.movment_state(data = data_frame)
            
            results = pd.concat([results,self.list_of_states, data_frame["body_cm"]], axis=1) 
            self.results_container.append(results)
            print(f"Progress: {int(((index+1)/self.data_n)*100)}%", "="*(index+1),">")
         

            

        self.results_gen =(gen for gen in self.results_container)

    def calculate_parameters(self, mode = "normal"): 
        self.results_parameters = pd.DataFrame(index = list(range(len(self.results_container))), columns = ["Rats_ID", "Time total [s]", "Time open total[s]", "Time open total[%]", "Time close total[s]", "Time close total[%]", "Time center total [s]", "Time center total [%]", "Time total_bin1 [s]", "Time open total_bin1[s]", "Time open total_bin1[%]", "Time close total_bin1[s]", "Time close total_bin1[%]", "Time center total_bin1 [s]", "Time center total_bin1 [%]",  "Time total_bin2 [s]", "Time open total_bin2[s]", "Time open total_bin2[%]", "Time close total_bin2[s]", "Time close total_bin2[%]", "Time center total_bin2 [s]", "Time center total_bin2 [%]", "Time total_bin3 [s]", "Time open total_bin3[s]", "Time open total_bin3[%]", "Time close total_bin3[s]", "Time close total_bin3[%]", "Time center total_bin3 [s]", "Time center total_bin3 [%]", "Time to first open [s]", "Time to first center [s]", "Min dist open total [cm]", "Min dist open_bin1 [cm]", "Min dist open_bin2 cm]", "Min dist open_bin3 [cm]", "Episodes of mobility_open_total [n]", "Episodes of immobility_open_total [n]", "Episodes of mobility_close_total [n]", "Episodes of immobility_close_total [n]", "Episodes of mobility_open_total_bin1 [n]", "Episodes of immobility_open_total_bin1 [n]", "Episodes of mobility_close_total_bin1 [n]", "Episodes of immobility_close_total_bin1 [n]", "Episodes of mobility_open_total_bin2 [n]", "Episodes of immobility_open_total_bin2 [n]", "Episodes of mobility_close_total_bin2 [n]", "Episodes of immobility_close_total_bin2 [n]", "Episodes of mobility_open_total_bin3 [n]", "Episodes of immobility_open_total_bin3 [n]", "Episodes of mobility_close_total_bin3 [n]", "Episodes of immobility_close_total_bin3 [n]", "Velocity mobility_total [cm/s]", "Velocity mobility_total_open [cm/s]", "Velocity immobility_total_open [cm/s]", "Velocity mobility_total_close [cm/s]", "Velocity mobility_total_bin1 [cm/s]", "Velocity mobility_total_open_bin1 [cm/s]", "Velocity immobility_total_open_bin1 [cm/s]", "Velocity mobility_total_close_bin1 [cm/s]", "Velocity mobility_total_bin2 [cm/s]", "Velocity mobility_total_open_bin2 [cm/s]", "Velocity immobility_total_open_bin2 [cm/s]", "Velocity mobility_total_close_bin2 [cm/s]", "Velocity mobility_total_bin3 [cm/s]", "Velocity mobility_total_open_bin3 [cm/s]", "Velocity immobility_total_open_bin3 [cm/s]", "Velocity mobility_total_close_bin3 [cm/s]", "Distance mobility_total [cm]", "Distance mobility_total_open [cm]", "Distance immobility_total_open [cm]", "Distance mobility_total_close [cm]", "Distance mobility_total_bin1 [cm]", "Distance mobility_total_open_bin1 [cm]", "Distance immobility_total_open_bin1 [cm]", "Distance mobility_total_close_bin1 [cm]", "Distance mobility_total_bin2 [cm]", "Distance mobility_total_open_bin2 [cm]", "Distance immobility_total_open_bin2 [cm]", "Distance mobility_total_close_bin2 [cm]", "Distance mobility_total_bin3 [cm]", "Distance mobility_total_open_bin3 [cm]", "Distance immobility_total_open_bin3 [cm]", "Distance mobility_total_close_bin3 [cm]", "Entry_open_total", "Entry_open_total_bin1", "Entry_open_total_bin2", "Entry_open_total_bin3", "Entry_center_total", "Entry_center_total_bin1", "Entry_center_total_bin2", "Entry_center_total_bin3"])
        self.graph_data_container_all = []
        for index, data_frame in enumerate(self.results_gen):
            results_container = []
            graph_data_container = []
            if mode == "normal":
                index_list = data_frame.index.tolist()
                index_list_3 = np.array_split(index_list, 3)
            elif mode == "GD_666":
                 index_list_3 = []
                 index_list_3.append([0, min(self.opto_duration[index])])
                 index_list_3.append([min(self.opto_duration[index]), max(self.opto_duration[index])])

                 index_list_3.append([max(self.opto_duration[index]), len(data_frame)-4])
            
            
            #time
            time_total = round(len(data_frame) / self.fps)
            time_open_total = round(len(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "Location"]) / self.fps)
            self.time_open_total_comulative =  data_frame["Location"].apply(lambda x: 1 if x == "NE_OPEN" or x == "SW_OPEN" else 0).tolist()
            self.time_open_total_comulative = [round(frame/self.fps, 3) if frame != 0 else frame for frame in self.time_open_total_comulative]
            self.time_open_total_comulative= np.cumsum(self.time_open_total_comulative)
            
            time_open_total_pr =  round((time_open_total/time_total * 100), 2)
            time_close_total = round(len(data_frame.loc[(data_frame["Location"] == "SE_CLOSE") | (data_frame["Location"] == "NW_CLOSE"), "Location"]) / self.fps)
            self.time_close_total_comulative =  data_frame["Location"].apply(lambda x: 1 if x == "NW_CLOSE" or x == "SE_CLOSE" else 0).tolist() 
            self.time_close_total_comulative = [round(frame/self.fps, 3) if frame != 0 else frame for frame in self.time_close_total_comulative]
            self.time_close_total_comulative = np.cumsum(self.time_close_total_comulative)
            
            time_close_total_pr =  round((time_close_total/time_total * 100), 2)
            time_center_total = round(len(data_frame.loc[(data_frame["Location"] == "center"), "Location"]) / self.fps)
            time_center_total_pr =  round((time_center_total/time_total * 100), 2)
            list_to_append = [time_total, time_open_total, time_open_total_pr, time_close_total, time_close_total_pr, time_center_total, time_center_total_pr]
            [results_container.append(ii) for ii in list_to_append]

            #time bins
            for i in range(3):
                df_bin = data_frame.iloc[min(index_list_3[i]): max(index_list_3[i]),:]
                time_total = round(len(df_bin) / self.fps)
                time_open_total = round(len(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "Location"]) / self.fps)
                time_open_total_pr =  round((time_open_total/time_total * 100), 2)
                time_close_total = round(len(df_bin.loc[(df_bin["Location"] == "SE_CLOSE") | (df_bin["Location"] == "NW_CLOSE"), "Location"]) / self.fps)
                time_close_total_pr =  round((time_close_total/time_total * 100), 2)
                time_center_total = round(len(df_bin.loc[(df_bin["Location"] == "center"), "Location"]) / self.fps)
                time_center_total_pr =  round((time_center_total/time_total * 100), 2)
                list_to_append = [time_total, time_open_total, time_open_total_pr, time_close_total, time_close_total_pr, time_center_total, time_center_total_pr]
                [results_container.append(iii) for iii in list_to_append]
            # time to first open/close
            results_container.append(self.time_to_open)
            
            results_container.append(self.time_to_close)
            # min_open_CM
            results_container.append(self.max_open_list[index])
            
            results_container.append(self.max_open_list_bin1[index])
            
            results_container.append(self.max_open_list_bin2[index])
            
            results_container.append(self.max_open_list_bin3[index])
            # episodes of imm/mobility
            df_open_total =  data_frame.loc[((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), :]
            episodes_mom_open, episodes_immo_open = self.number_of_movement(data = df_open_total)
            df_close_total =  data_frame.loc[((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), :]
            episodes_mom_close, episodes_immo_close = self.number_of_movement(data = df_close_total)
            list_to_append = [episodes_mom_open, episodes_immo_open, episodes_mom_close, episodes_immo_close]
            [results_container.append(iiii) for iiii in list_to_append]

            for i in range(3):
                df_bin_1 = data_frame.iloc[min(index_list_3[i]): max(index_list_3[i]),:]
                df_bin_1_open =  df_bin_1.loc[((df_bin_1["Location"] == "NE_OPEN") | (df_bin_1["Location"] == "SW_OPEN")), :]
                episodes_mom_bin1_open, episodes_immo_bin1_open = self.number_of_movement(data = df_bin_1_open)
                df_bin_1_close =  df_bin_1.loc[(df_bin_1["Location"] == "NW_CLOSE") | (df_bin_1["Location"] == "SE_CLOSE"), :]
                episodes_mom_bin1_close, episodes_immo_bin1_close = self.number_of_movement(data = df_bin_1_close)
                list_to_append = [episodes_mom_bin1_open, episodes_immo_bin1_open, episodes_mom_bin1_close, episodes_immo_bin1_close]
                [results_container.append(iiiii) for iiiii in list_to_append]

            # anlysis of movement speed
            
            total_speed_mo = round(sum(data_frame.loc[(data_frame["movment_state"] == 1), "body_cm"]) / (len(data_frame.loc[data_frame["movment_state"] == 1, "movment_state"])/self.fps),2)
            total_speed_mo_close = round(sum(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), "body_cm"]) / (len(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), "movment_state"]) / self.fps),2)
            total_speed_imm_mo_open = round(sum(data_frame.loc[((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), "body_cm"]) / (len(data_frame.loc[((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), "movment_state"]) / self.fps),2)

            if len(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "movment_state"]) == 0:
                total_speed_mo_open = "None"
            else:
                total_speed_mo_open = round(sum(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "body_cm"]) / (len(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "movment_state"]) / self.fps),2)
            
            if len(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "movment_state"]) == 0:
                total_speed_imm_mo_open =  "None"
            else:
                total_speed_imm_mo_open =  round(sum(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "body_cm"]) / (len(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "movment_state"]) / self.fps),2)
            
            list_to_append = [total_speed_mo, total_speed_mo_open, total_speed_imm_mo_open, total_speed_mo_close]
            [results_container.append(j) for j in list_to_append]


            for i in range(3):
                df_bin = data_frame.iloc[min(index_list_3[i]): max(index_list_3[i]),:]
                total_speed_mo = round(sum(df_bin.loc[(df_bin["movment_state"] == 1), "body_cm"]) / (len(df_bin.loc[df_bin["movment_state"] == 1, "movment_state"])/self.fps),2)
                total_speed_mo_close = round(sum(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NW_CLOSE") | (df_bin["Location"] == "SE_CLOSE")), "body_cm"]) / (len(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NW_CLOSE") | (df_bin["Location"] == "SE_CLOSE")), "movment_state"]) / self.fps),2)
                
                
                
                if len(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN")), "movment_state"]) == 0:
                    total_speed_mo_open = "None"
                else:
                    total_speed_mo_open = round(sum(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN")), "body_cm"]) / (len(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN")), "movment_state"]) / self.fps),2)

                if len(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "movment_state"]) == 0:
                    total_speed_imm_mo_open =  "None"
                else:
                    total_speed_imm_mo_open =  round(sum(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "body_cm"]) / (len(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "movment_state"]) / self.fps),2)
                
                list_to_append = [total_speed_mo, total_speed_mo_open, total_speed_imm_mo_open, total_speed_mo_close]
                [results_container.append(jj) for jj in list_to_append]
            
            # anlysis of distance 
            total_distance_mo = float(round(sum(data_frame.loc[(data_frame["movment_state"] == 1), "body_cm"]) ,2))
            self.time_open_total_comulative = [round(frame/self.fps, 3) if frame != 0 else frame for frame in self.time_open_total_comulative]
            self.body_cm_list_v23 = data_frame.loc[:, "body_cm"].tolist()
            self.total_distance_mo_cumulative = [self.body_cm_list_v23[xx] if data_frame["movment_state"][xx] == 1 else 0 for xx in range(len(self.body_cm_list_v23))]
            self.total_distance_mo_cumulative = np.cumsum(self.total_distance_mo_cumulative)
            
            total_distance_mo_close = float(round(sum(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NW_CLOSE") | (data_frame["Location"] == "SE_CLOSE")), "body_cm"]),2))
            self.total_distance_mo_close_cumulative = [self.body_cm_list_v23[xx] if data_frame["movment_state"][xx] == 1 and (data_frame["Location"][xx] == "NW_CLOSE" or data_frame["Location"][xx] == "SE_CLOSE")  else 0 for xx in range(len(self.body_cm_list_v23))]
            self.total_distance_mo_close_cumulative = np.cumsum(self.total_distance_mo_close_cumulative)
            
            if len(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "movment_state"]) == 0:
                    total_distance_mo_open = "None"
                    self.total_distance_mo_open_cumulative_v2 = "None"
            else:
                    total_distance_mo_open = float(round(sum(data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "body_cm"]),2))
                    self.total_distance_mo_open_cumulative =  data_frame.loc[(data_frame["movment_state"] == 1) & ((data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN")), "body_cm"].tolist()
                    self.total_distance_mo_open_cumulative = [self.body_cm_list_v23[xx] if data_frame["movment_state"][xx] == 1 and (data_frame["Location"][xx] == "NE_OPEN" or data_frame["Location"][xx] == "SW_OPEN")  else 0 for xx in range(len(self.body_cm_list_v23))]
                    self.total_distance_mo_open_cumulative_v2 = np.cumsum(self.total_distance_mo_open_cumulative)
                    assert len(self.total_distance_mo_open_cumulative_v2) == len(self.total_distance_mo_open_cumulative)
                    
            if len(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "movment_state"]) == 0:
                    total_distance_imm_mo_open = "None"
            else:
                    total_distance_imm_mo_open = float(round(sum(data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "body_cm"]),2))
                    self.total_distance_imm_mo_open = data_frame.loc[(data_frame["Location"] == "NE_OPEN") | (data_frame["Location"] == "SW_OPEN"), "body_cm"].tolist()
                    self.total_distance_imm_mo_open_cumulative = [self.body_cm_list_v23[xx] if data_frame["Location"][xx] == "NE_OPEN" or data_frame["Location"][xx] == "SW_OPEN"  else 0 for xx in range(len(self.body_cm_list_v23))]
                    self.total_distance_imm_mo_open_cumulative = np.cumsum(self.total_distance_imm_mo_open_cumulative)
            
            list_to_append = [total_distance_mo, total_distance_mo_open, total_distance_imm_mo_open, total_distance_mo_close]
            [results_container.append(jjj) for jjj in list_to_append]
            
            for i in range(3):
                df_bin = data_frame.iloc[min(index_list_3[i]): max(index_list_3[i]),:]
                total_distance_mo = float(round(sum(df_bin.loc[(df_bin["movment_state"] == 1), "body_cm"]) ,2))
                total_distance_mo_close = float(round(sum(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NW_CLOSE") | (df_bin["Location"] == "SE_CLOSE")), "body_cm"]),2))
                
                if len(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN")), "movment_state"]) == 0:
                    total_distance_mo_open = "None"
                else:
                    total_distance_mo_open = float(round(sum(df_bin.loc[(df_bin["movment_state"] == 1) & ((df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN")), "body_cm"]),2))
                
                if len(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "movment_state"]) == 0:
                    total_distance_imm_mo_open =  "None"
                else:
                    total_distance_imm_mo_open =  float(round(sum(df_bin.loc[(df_bin["Location"] == "NE_OPEN") | (df_bin["Location"] == "SW_OPEN"), "body_cm"]),2))

                list_to_append = [total_distance_mo, total_distance_mo_open, total_distance_imm_mo_open, total_distance_mo_close]
                [results_container.append(jjjj) for jjjj in list_to_append]
            
            # entrances 
            results_container.append(self.entry_open_list[index])
            results_container.append(self.entry_open_list_bin1[index])
            results_container.append(self.entry_open_list_bin2[index])
            results_container.append(self.entry_open_list_bin3[index])
            results_container.append(self.entry_center_list[index])
            results_container.append(self.entry_center_list_bin1[index])
            results_container.append(self.entry_center_list_bin2[index])
            results_container.append(self.entry_center_list_bin3[index])
            results_container.insert(0, self.rat_id[index])
            #append results to df
            
            self.results_parameters.iloc[index, :] = results_container
            #graph
            self.graph_data_container_all.append([self.time_open_total_comulative, self.time_close_total_comulative, self.total_distance_mo_cumulative, self.total_distance_mo_open_cumulative_v2, self.total_distance_mo_close_cumulative, self.total_distance_imm_mo_open_cumulative])
    
    def backward(self):
        
        if self.current_cm_NE < self.current_cm_NE_BODY:
             if self.backward_type_2 == "started" and self.duration > 0:
                    self.backward_type_2 = "stop"
                    self.stop_postion = self.current_poly
                    self.previous_cm_NE = "unknown"
                    self.previous_cm_NE_BODY = "unknown"
                    self.punishment = 0
                    stop_time = round(self.current_time/self.fps,2)
                    if self.current_time <  max(self.brodcasting_index_list_3[0]):
                         bin_stage = "bin_1"
                    elif self.current_time < max(self.brodcasting_index_list_3[1]):
                         bin_stage = "bin_2"
                    elif self.current_time < max(self.brodcasting_index_list_3[2]):
                         bin_stage = "bin_3"
                    self.backward_episodes.append([ round(self.duration/int(self.fps),2), self.start_postion, self.stop_postion, bin_stage])
                    self.duration = 0
                    self.backward_controler = False
                    self.start_time = 0
             elif self.backward_type_2 == "started":
                  self.backward_type_2 = "stop"
                  self.previous_cm_NE = "unknown"
                  self.previous_cm_NE_BODY = "unknown"
             
             
             if self.previous_cm_NE == "unknown":
                  self.previous_cm_NE = self.current_cm_NE
                  self.previous_cm_NE_BODY = self.current_cm_NE_BODY
                  self.backward_type_1 = "started"
             
             elif self.backward_type_1 =="started":
                  
                  head_delta = self.current_cm_NE -self.previous_cm_NE 
                  body_delta = self.current_cm_NE_BODY - self.previous_cm_NE_BODY
                  self.previous_cm_NE = self.current_cm_NE
                  self.previous_cm_NE_BODY = self.current_cm_NE_BODY
                  
                  if head_delta >= 0.2 or body_delta >= 0.2:
                       if self.duration == 0:
                            self.start_postion = self.current_poly
                            self.start_time = round(self.current_time/self.fps,2)
                       self.duration +=1
                       self.punishment = 0
                       self.backward_controler = True 
                       
                  elif (head_delta < -0.1 or body_delta < -0.1 or self.punishment > int(self.fps)) and self.backward_controler:
                       self.backward_type_1 = "stop"
                       self.stop_postion = self.current_poly
                       self.previous_cm_NE = "unknown"
                       self.previous_cm_NE_BODY = "unknown"
                       self.punishment = 0
                       
                       if self.current_time <  max(self.brodcasting_index_list_3[0]):
                            bin_stage = "bin_1"
                       elif self.current_time < max(self.brodcasting_index_list_3[1]):
                            bin_stage = "bin_2"
                       elif self.current_time < max(self.brodcasting_index_list_3[2]):
                            bin_stage = "bin_3"
                       self.backward_episodes.append([ round(self.duration/int(self.fps),2), self.start_postion, self.stop_postion, bin_stage])
                       self.duration = 0
                       self.backward_controler = False
                       self.start_time = 0
                       
                  
                  elif (head_delta >= -0.1 or body_delta >= -0.1) and self.backward_controler:
                       self.duration +=1 
                       self.punishment +=1
                      
                  
        elif self.current_cm_NE > self.current_cm_NE_BODY:
             
             if self.backward_type_1 == "started" and self.duration > 0:
                    self.backward_type_1 = "stop"
                    self.stop_postion = self.current_poly
                    self.previous_cm_NE = "unknown"
                    self.previous_cm_NE_BODY = "unknown"
                    self.punishment = 0
                    stop_time = round(self.current_time/self.fps,2)
                    if self.current_time <  max(self.brodcasting_index_list_3[0]):
                        bin_stage = "bin_1"
                    elif self.current_time < max(self.brodcasting_index_list_3[1]):
                        bin_stage = "bin_2"
                    elif self.current_time < max(self.brodcasting_index_list_3[2]):
                        bin_stage = "bin_3"
                    self.backward_episodes.append([ round(self.duration/int(self.fps),2), self.start_postion, self.stop_postion, bin_stage])
                    self.duration = 0
                    self.backward_controler = False
                    self.start_time = 0
             elif self.backward_type_1 == "started":
                  self.backward_type_1 = "stop"
                  self.previous_cm_NE = "unknown"
                  self.previous_cm_NE_BODY = "unknown"
             
             if self.previous_cm_NE == "unknown":
                  self.previous_cm_NE = self.current_cm_NE
                  self.previous_cm_NE_BODY = self.current_cm_NE_BODY
                  self.backward_type_2 = "started"
             elif self.backward_type_2 == "started":
                  head_delta = self.previous_cm_NE - self.current_cm_NE 
                  body_delta = self.previous_cm_NE_BODY - self.current_cm_NE_BODY
                  self.previous_cm_NE = self.current_cm_NE
                  self.previous_cm_NE_BODY = self.current_cm_NE_BODY
                  
                  if head_delta >= 0.2 or body_delta >= 0.2:
                       if self.duration == 0:
                            self.start_postion = self.current_poly
                            self.start_time = round(self.current_time/self.fps,2)
                       self.duration +=1
                       self.punishment = 0
                       self.backward_controler =  True
                       
                  elif (head_delta < -0.1 or body_delta < -0.1 or self.punishment > int(self.fps)) and self.backward_controler:
                       self.backward_type_2 = "stop"
                       self.stop_postion = self.current_poly
                       self.previous_cm_NE = "unknown"
                       self.previous_cm_NE_BODY = "unknown"
                       self.punishment = 0
                       stop_time = round(self.current_time/self.fps,2)
                       if self.current_time <  max(self.brodcasting_index_list_3[0]):
                            bin_stage = "bin_1"
                       elif self.current_time < max(self.brodcasting_index_list_3[1]):
                            bin_stage = "bin_2"
                       elif self.current_time < max(self.brodcasting_index_list_3[2]):
                            bin_stage = "bin_3"
                       self.backward_episodes.append([ round(self.duration/int(self.fps),2), self.start_postion, self.stop_postion, bin_stage])
                       self.duration = 0 
                       self.start_time = 0
                       self.backward_controler = False
                       
                  elif (head_delta >= -0.1 or body_delta >= -0.1) and self.backward_controler:
                       self.duration +=1
                       self.punishment +=1
    
    def save_results(self):
         
         self.sf = easygui.diropenbox(msg = "Select folder for a save location", title = "Typical window")                 
         for index, sub_list in enumerate(self.episodes_head_center_body_close_all):
              results_head_center_body_close_all = pd.DataFrame(index = list(range(len(sub_list))), columns = ["Rat_id", "Bin", "Time [s]"])
              for index_v1, base_list in enumerate(sub_list):
                   base_list.insert(0,self.rat_id[index])
                   results_head_center_body_close_all.iloc[index_v1, :] = base_list
              
              results_head_center_body_close_all.to_excel(self.sf + "//" + f"{self.rat_id[index]}_results_head_center_body_close.xlsx")
         
         for index, sub_list in enumerate(self.backward_episodes_all):
              results_backward_episodes_all = pd.DataFrame(index = list(range(len(sub_list))), columns = ["Rat_id", "Time [s]", "Start_in", "Stop_in", "Bin"])
              for index_v1, base_list in enumerate(sub_list):
                   base_list.insert(0,self.rat_id[index])
                   results_backward_episodes_all.iloc[index_v1, :] = base_list

              results_backward_episodes_all.to_excel(self.sf + "//" + f"{self.rat_id[index]}_results_backward_episodes.xlsx")

         self.results_parameters.to_excel(self.sf + "//" + "main_results_all.xlsx")
         self.graphs()
    
    def graphs(self):
         # Cumulative plots
         for index, data in enumerate(self.graph_data_container_all):
              y_title = ["Cumulative time [s]", "Cumulative time [s]", "Cumulative distance [cm]", "Cumulative distance [cm]", "Cumulative distance [cm]", "Cumulative distance [cm]"]
              fig_title = ["Time_in_open_arm_of_EPM_test_by_", "Time_in_close_arm_of_EPM_test_by_", "Distance_travelled_during_mobility_in_EPM_test_by_", "Distance_travelled_during mobility_in_open_arm_of_EPM_by_", "Distance_travelled_during_mobility_in_close_arm_of_EPM_by_", "Distance_travelled_during_mobility_and_immobility_in_open_arm_of_EPM_test_by_"]
              for indexx in range(len(data)):
                   # plots
                   plt.style.use('seaborn')
                   plt.plot(range(len(data[indexx])), data[indexx])
                   plt.xlabel("Frames")
                   plt.ylabel(y_title[indexx])
                   plt.title(f"{fig_title[indexx]} {self.rat_id[index]}")
                   plt.savefig(self.sf + "//" + f"{fig_title[indexx]} {self.rat_id[index]}.svg")
                   plt.clf()
         # Distance travelled plots
         for index, data in enumerate(self.travelled_plot):
              fig, axs = plt.subplots(2)
              plt.style.use('classic')
              axs[0].plot(data[0], data[1], "-r", alpha = 0.6, )
              axs[0].imshow(self.frames)
              axs[0].set_xlabel("X position in pixel")
              axs[0].set_ylabel("Y position in pixel")
              axs[0].set_title(f"Tract tracing of {self.rat_id[index]} in EPM test")
              axs[0].xaxis.set_ticks([])
              axs[0].yaxis.set_ticks([])
              
              sns.kdeplot(x = data[0],y = data[1], shade=True, shade_lowest=False, bw_adjust = 0.15, cmap = "rainbow", cbar = True, ax= axs[1])
              axs[1].imshow(self.frames)
              axs[1].set_xlabel("X position in pixel")
              axs[1].set_ylabel("Y position in pixel")
              axs[1].set_title(f"Density map of {self.rat_id[index]} in EPM test")
              axs[1].xaxis.set_ticks([])
              axs[1].yaxis.set_ticks([])
              plt.savefig(self.sf + "//" + f"Tract tracing_of_ {self.rat_id[index]}.svg")
              plt.cla()

epm = EPM_object()
epm.pre_processing(mode = "GD_666")
epm.get_position(mode = "GD_666")
epm.calculate_parameters()
epm.save_results()
