import PySimpleGUI as sg
import cv2
import numpy as np

"""
Demo program that displays a webcam using OpenCV
"""
namedict = {
"Ben Affleck" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/benaffleck.png",
"Beyonce" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/beyonce.png",
"Brianna Cuoco" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/briannacuoco.png",
"Casey Affleck" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/caseyaffleck.png",
"Dave Franco" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/davefranco.png",
"Haley Duff" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/haleyduff.png",
"Hilary" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/hilary.png",
"James Franco" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/jamesfranco.png",
"Kaley Cuoco" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/kaleycuoco.png",
"Solange" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/solange.png",
"Solange" : "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/solange.png",
}

def main():

    sg.theme('BrownBlue')

    # define the window layout
    videofeed = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='video'),sg.Image(filename='', key='snap')],
               [sg.Button('Snap', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]
    message=[
            [sg.Text('You looklike', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Image(filename=fname, key='image')]
            ]
    layout = [
        [
            sg.Column(videofeed, element_justification="c"),
            sg.VSeperator(),
            sg.Column(message, element_justification="c"),
        ]
    ]
    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #

    recording = False
    cap = cv2.VideoCapture(0)
    recording = True
    while True:

        event, values = window.read(timeout=20)

        # ret, frame = cap.read()
        # video_start = cv2.imencode('.png', frame)[1].tobytes()
        # window['video'].update(data=video_start)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Snap':
            #recording = False
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            cv2.imwrite('snapshot.png', frame)
            window['snap'].update(filename='snapshot.png')

        if recording:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(300,250),interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()# ditto
            window['video'].update(data=imgbytes)


main()