import argparse
import imutils
from imutils import paths
import time
import dlib
import cv2
import os

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)

imagePaths = list(paths.list_images(args['image']))
for (i,imagePath) in enumerate(imagePaths):
	#print(imagePath)
	file = imagePath.split(os.path.sep)[-1]
	name = imagePath.split(os.path.sep)[-2]
	name = name + "_Cropped"

	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=256)
	#print(image.shape)

	image2 = image
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# perform face detection using dlib's face detector
	start = time.time()
	print("[INFO[ performing face detection with dlib...")
	rects = detector(rgb, args["upsample"])
	end = time.time()
	print("[INFO] face detection took {:.4f} seconds".format(end - start))

	# convert the resulting dlib rectangle objects to bounding boxes,
	# then ensure the bounding boxes are all within the bounds of the
	# input image
	boxes = [convert_and_trim_bb(image, r) for r in rects]
	# loop over the bounding boxes
	for (x, y, w, h) in boxes:
		# draw the bounding box on our image
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cropped = image[y:y+h, x:x+w]
	cropped_path = os.path.join("image_data/cropped_image/",name)
	cropped_path = os.path.join(cropped_path, file)
	#print(cropped_path)
	# show the output image
	#cv2.imshow("Output", image)
	#cv2.imshow("Cropped", cropped)
	cv2.imwrite(os.path.join(os.path.join("image_data/cropped_image/",name),file), cropped)
	#cv2.waitKey(0)



