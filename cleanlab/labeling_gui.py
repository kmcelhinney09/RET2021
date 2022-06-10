import os
import pandas as pd
import PySimpleGUI as sg
import shutil

label_error_figs = sorted(os.listdir("./label_error_figs"))
num_error_figs = len(label_error_figs)

original_labels_df = pd.read_csv("/home/nthom/Documents/datasets/CelebA/Anno/list_attr_celeba.txt", skiprows=1, sep=" ")
original_labels_df.set_index("image_name", inplace=True)

labels_df = pd.read_csv("./list_attr_celeba_relabeled.txt", sep=" ", skiprows=1)
labels_df.set_index("image_name", inplace=True)

save_df = pd.read_csv("./relabel_stats.csv")
count = save_df.loc[0][0]
num_labels_changed = save_df.loc[0][1]


warning_message = ""

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [ [sg.Text('Please agree or disagree with the following label correction:')],
            [sg.Image(filename=f"", key='fig')],
            [sg.Button('Label As Positive'), sg.Button('Label As Negative'), sg.Button('Previous Image'), sg.Button('Exit Labeler')],
            [sg.Text(f'Image: '), sg.Text(f"{count+1}", key="curr_img"), sg.Text(f'Of: '), sg.Text(f"{num_error_figs}", key="total"), sg.Text(f'Info/Warning: '), sg.Text(f'', key="warn")],
            ]

# Create the Window
window = sg.Window('Attribute Labeler', layout)
# Event Loop to process "events" and get the "values" of the inputs
window.finalize()

while True:
    current_fig_name = label_error_figs[count]
    window["fig"].update(filename=f"./label_error_figs/{current_fig_name}")

    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit Labeler': # if user closes window or clicks cancel
        break

    if event == "Previous Image":
        if count == 0:
            window["warn"].update("Cannot go to previous image because the current image is the first image.")
        else:
            count -= 1
            window["curr_img"].update(f"{count+1}")

            current_image_name = label_error_figs[count][-10:-4]
            current_attribute = label_error_figs[count].split("_")[0]

            if labels_df.loc[f"{current_image_name}.jpg", current_attribute] != original_labels_df.loc[f"{current_image_name}.jpg", current_attribute]:
                labels_df.loc[f"{current_image_name}.jpg", current_attribute] = original_labels_df.loc[f"{current_image_name}.jpg", current_attribute]
                num_labels_changed -= 1

    elif event == 'Label As Positive': # if user closes window or clicks cancel
        current_image_name = label_error_figs[count][-10:-4]
        current_attribute = label_error_figs[count].split("_")[0]

        if original_labels_df.loc[f"{current_image_name}.jpg", current_attribute] != 1:
            labels_df[current_image_name, current_attribute] = 1
            num_labels_changed += 1

        window["warn"].update("")
        count += 1
        window["curr_img"].update(f"{count + 1}")
        continue

    elif event == 'Label As Negative':
        current_image_name = label_error_figs[count][-10:-4]
        current_attribute = label_error_figs[count].split("_")[0]

        if original_labels_df.loc[f"{current_image_name}.jpg", current_attribute] != -1:
            labels_df[current_image_name, current_attribute] = -1
            num_labels_changed += 1
            os

        window["warn"].update("")
        count += 1
        window["curr_img"].update(f"{count + 1}")
        continue

window["warn"].update("Saving progess to: ./list_attr_celeba_relabeled.txt and statistics to: ./relabel_stats.csv")

labels_df.reset_index(inplace=True)
labels_df.to_csv("./list_attr_celeba_relabeled_test.txt", index=False, sep=" ")

save_df.loc[0] = [count, num_labels_changed]
save_df.to_csv("./relabel_stats.csv", index=False)

window.close()