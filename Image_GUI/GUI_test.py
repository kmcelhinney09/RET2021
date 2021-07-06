import PySimpleGUI as sg
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

sg.theme('dark green')


layout = [
    [sg.Text("Face_Recogniser", justification='center', font=("Times",40),size=(25,2))],
    [sg.Text("Number of images to take",font=("Times"),), sg.Input()],
    [sg.Text("Name for file",font=("Times")),sg.Input()],
    [sg.Button("Sample",size=(25,1)),sg.Button("Training",size=(25,1)),sg.Button("Testing",size=(25,1)),
     sg.Button("Quit",size=(25,1))],

]

window = sg.Window("Face_Recogniser", layout,element_justification='c')
# event loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
window.close()
