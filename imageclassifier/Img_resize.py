from imutils import paths
import cv2
import os
croppedPaths = list(paths.list_images('image_data/cropped_image/Part_2'))
#print(croppedPaths)

for (i,croppedPath) in enumerate(croppedPaths):
    file2 = croppedPath.split(os.path.sep)[-1]
    name2 = croppedPath.split(os.path.sep)[-2]
    name = name2[:-8]
    print(name)

    cropped_image = cv2.imread(croppedPath)
    dim = (128,128)
    cropped_image = cv2.resize(cropped_image,dim,interpolation=cv2.INTER_AREA)
    print(cropped_image.shape)
    cv2.imwrite(os.path.join(os.path.join("image_data/cropped_image/Part_2", name), file2), cropped_image)
