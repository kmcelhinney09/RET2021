# image_browser.py
import glob
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk

images_to_clean = []


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def parse_dir(path):
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

        file_name = path_split[0] + "/" + path_split[-1]

        window["-Output-"].update(file_name)
    except:
        print(f"Unable to open {path}!")


def main():
    elements = [
        [sg.Text(size=(100, 1), key="-Output-")],
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
        ],
        [
            sg.Text("Mark Folder as Complete and Move"),
        ],
        [
            sg.Input(size=(25, 1), enable_events=True, key="directory_complete"),
            sg.Button("Set File Path", key="Path"), sg.Button("Done")

        ]
    ]

    window = sg.Window("Image Viewer", elements, size=(800, 800), element_justification='c')
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
        if event == "Path":
            window["directory_complete"].update(images[location])
        if event == "Done":
            original_file_path = values["directory_complete"]
            path_list = splitall(original_file_path)
            file_name = path_list[-1]
            celeb_folder = path_list[-2]
            completed_folder = "completed"
            top_dir = path_list[-4]
            folder_path = os.path.join(os.path.join(top_dir, path_list[-3]), celeb_folder)
            # make completed folder
            completed_path = os.path.join(top_dir, completed_folder)
            if not os.path.exists(completed_path):
                os.makedirs(completed_path)
            rename_folder = os.path.join(completed_path, celeb_folder)
            os.rename(folder_path, rename_folder)

    window.close()


if __name__ == "__main__":
    main()
