# image_browser.py
import glob
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk

images_to_clean = []
def parse_dir (path):
    dirs = glob.glob(f'{path}/*')
    dirs.sort()
    return dirs

def parse_folder(path):
    images = glob.glob(f'{path}/*')
    images.sort()
    for image in images:
        images_to_clean.append(image)
    return images_to_clean


def load_image(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((500, 500))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
        path_split = os.path.split(path)
        celeb_name = os.path.split(path_split[0])

        file_name =celeb_name[1] + "/" + path_split[-1]

        window["-Output-"].update(file_name)
    except:
        print(f"Unable to open {path}!")

def main():
    elements = [
        [sg.Text(size=(30,1),key="-Output-")],
        [sg.Image(key="image")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), enable_events=True, key="file"),
            sg.FolderBrowse(),
        ],
        [
            sg.Button("Previous"),
            sg.Button("Delete"),
            sg.Button("Next")
        ]
    ]

    window = sg.Window("Image Viewer", elements, size=(800, 800))
    images = []
    location = 0

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "file":
            dirs = parse_dir(values["file"])
            if dirs:
                for dir in dirs:
                    images = parse_folder(dir)
                    if images:
                        load_image(images[0], window)
        if event == "Next" and images:
            if location == len(images) - 1:
                location = 0
            else:
                location += 1
            load_image(images[location], window)
        if event == "Delete" and images:
            os.remove(images[location])
            location += 1
            load_image(images[location], window)
        if event == "Previous" and images:
            if location == 0:
                location = len(images) - 1
            else:
                location -= 1
            load_image(images[location], window)

    window.close()


if __name__ == "__main__":
    main()