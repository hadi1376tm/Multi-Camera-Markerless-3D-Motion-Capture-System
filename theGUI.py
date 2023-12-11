# existing_parameters
# Done1#

import tkinter as tk
from tkinter import Tk, Label, Button, Frame, Listbox, Entry, filedialog

from PIL import Image, ImageTk
import pickle
import datetime
from pathlib import Path
import sys



# Thinker window caller to select joints to calculate their angle
def Select_joints_angle_calculator():
    root = tk.Tk()
    jointsSelect = JointsCheckboxWindow(root)
    root.mainloop()

    return jointsSelect.selectedRightJoints,jointsSelect.selectedLeftJoints

# Thinker GUI to select joints to calculate their angle

class JointsCheckboxWindow:
    def __init__(self, master):
        self.master = master
        self.selectedLeftJoints = []
        self.selectedRightJoints = []


        self.master.title("Angles")
        topFrame = Frame(self.master)
        topFrame.pack(side=tk.TOP)
        taskText = "Select angles to calculate"
        taskLabel = Label(topFrame, text=taskText, font="bold",pady= 5)
        taskLabel.pack()
        parametersFrame = Frame(self.master)
        parametersFrame.pack()




        # Add the image
        image = Image.open("bodyLine.jpg")
        image = image.resize((image.width*2, image.height*2), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        imageLabel = Label(parametersFrame, image=image)
        imageLabel.image = image
        imageLabel.pack()
        leftLable = Label(parametersFrame, text="Right", font=("Arial", 14), bg="white")
        leftLable.place(x=20, y=20)
        leftLable = Label(parametersFrame, text="Left", font=("Arial", 14), bg="white")
        leftLable.place(x=160, y=20)

        bottomFrame = Frame(self.master).pack(side=tk.BOTTOM)

        self.checkbox1_var = tk.IntVar()
        self.checkbox2_var = tk.IntVar()
        self.checkbox3_var = tk.IntVar()
        self.checkbox4_var = tk.IntVar()
        self.checkbox5_var = tk.IntVar()
        self.checkbox6_var = tk.IntVar()
        self.checkbox7_var = tk.IntVar()
        self.checkbox8_var = tk.IntVar()
        self.checkbox9_var = tk.IntVar()
        self.checkbox10_var = tk.IntVar()

        # left joints checkboxes
        self.checkbox1 = tk.Checkbutton(parametersFrame,variable=self.checkbox1_var, command=lambda: self.add_to_rightJoints_selected("right_shoulder"))
        self.checkbox1.place(x=58,y=100)


        self.checkbox2 = tk.Checkbutton(parametersFrame,variable=self.checkbox2_var, command=lambda: self.add_to_rightJoints_selected("right_elbow"))
        self.checkbox2.place(x=47,y=155)


        self.checkbox3 = tk.Checkbutton(parametersFrame,variable=self.checkbox3_var, command=lambda: self.add_to_rightJoints_selected("right_hip"))
        self.checkbox3.place(x=70,y=225)


        self.checkbox4 = tk.Checkbutton(parametersFrame,variable=self.checkbox4_var, command=lambda: self.add_to_rightJoints_selected("right_knee"))
        self.checkbox4.place(x=78,y=322)


        self.checkbox5 = tk.Checkbutton(parametersFrame,variable=self.checkbox5_var, command=lambda: self.add_to_rightJoints_selected("right_ankle"))
        self.checkbox5.place(x=75,y=422)




        # left joints checkboxes
        self.checkbox6 = tk.Checkbutton(parametersFrame,variable=self.checkbox6_var,command=lambda: self.add_to_leftJoints_selected("left_shoulder"))
        self.checkbox6.place(x=145,y=100)


        self.checkbox7 = tk.Checkbutton(parametersFrame,variable=self.checkbox7_var,command=lambda: self.add_to_leftJoints_selected("left_elbow"))
        self.checkbox7.place(x=160,y=155)

        self.checkbox8 = tk.Checkbutton(parametersFrame,variable=self.checkbox8_var,command=lambda: self.add_to_leftJoints_selected("left_hip"))
        self.checkbox8.place(x=140,y=225)


        self.checkbox9 = tk.Checkbutton(parametersFrame,variable=self.checkbox9_var,command=lambda: self.add_to_leftJoints_selected("left_knee"))
        self.checkbox9.place(x=133,y=322)

        self.checkbox10 = tk.Checkbutton(parametersFrame,variable=self.checkbox10_var,command=lambda: self.add_to_leftJoints_selected("left_ankle"))
        self.checkbox10.place(x=133,y=422)


        self.continue_button = tk.Button(bottomFrame, text="Continue", command=self.continue_button_click)
        self.continue_button.pack(side=tk.BOTTOM)

    def add_to_leftJoints_selected(self, option):
        if option not in self.selectedLeftJoints:
            self.selectedLeftJoints.append(option)
        else:
            self.selectedLeftJoints.remove(option)

    def add_to_rightJoints_selected(self, option):
        if option not in self.selectedRightJoints:
            self.selectedRightJoints.append(option)
        else:
            self.selectedRightJoints.remove(option)

    def continue_button_click(self):
        # print("-Selected joints:", self.selectedRightJoints, self.selectedLeftJoints)
        self.master.destroy()