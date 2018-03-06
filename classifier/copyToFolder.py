import os
import numpy as np
from shutil import copyfile

srcfold = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/Static Pose '
poseNumArray = [1, 2, 13, 14, 28, 29, 37, 38, 39, 40, 51, 52, 56, 57]

percentToTest = 10
depthOrDist = 'depth'

destfold = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/static_poses_' + depthOrDist + '/F1/'

for pose in poseNumArray:
    curFold = srcfold + str(pose) + '/' + depthOrDist + '/'
    ims = [im for im in os.listdir(curFold)]
    sampl = np.arange(1000)
    np.random.shuffle(sampl)
    train = sampl[:len(sampl) - int(len(sampl) * (1/percentToTest))]
    test = sampl[len(sampl) - int(len(sampl) * (1/percentToTest)):]

    pathFold = destfold + 'train/p' + str(pose) + '/'
    for imgsPath in os.listdir(pathFold):
        if os.path.isfile(pathFold + imgsPath):
            os.unlink(pathFold + imgsPath)

    pathFold = destfold + 'test/p' + str(pose) + '/'
    for imgsPath in os.listdir(pathFold):
        if os.path.isfile(pathFold + imgsPath):
            os.unlink(pathFold + imgsPath)

    print(curFold)

    for val in train:
        imName = ims[val]
        copyfile(curFold + imName, destfold + 'train/p' + str(pose) + '/' + imName)

    for val in test:
        imName = ims[val]
        copyfile(curFold + imName, destfold + 'test/p' + str(pose) + '/' + imName)
