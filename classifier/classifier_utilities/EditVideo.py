import numpy as np
import cv2 as cv
import os


folder_src = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/uncuted'
folder_dest = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/DB'
folder_trash_dest = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/uncuted'

cv.namedWindow('video')
cv.namedWindow('currentCut')

curr_first = 0
length_show = 10
curr_last = curr_first + length_show

#video information
current_gesture = 0  # PX
current_video = 0    # EX
current_video_frame_cont = length_show

# images to show in cut window
show_top = []
show_botton = []

for idx, uncut_vid_path in enumerate(os.listdir(folder_src)):
    uncuted_video_folder = sorted(os.listdir(os.path.join(folder_src,
                                                          uncut_vid_path)),
                                  key=len)
    uncuted_video_folder = [os.path.join(folder_src, uncut_vid_path, l) for l in uncuted_video_folder]
    size_total = len(uncuted_video_folder)

    cutting = True
    open_cut = False

    for idx_rng in range(0, 5):
        show_top.append(cv.imread(uncuted_video_folder[idx_rng]))
    for idx_rng in range(5, 10):
        show_botton.append(cv.imread(uncuted_video_folder[idx_rng]))

    loading = True
    while cutting:

        if loading:
            top_img = show_top[0]
            for idx_top_img in range(1, len(show_top)):
                top_img = np.concatenate((top_img, show_top[idx_top_img]), axis=1)

            bot_img = show_botton[0]
            for idx_top_img in range(1, len(show_top)):
                bot_img = np.concatenate((bot_img, show_botton[idx_top_img]), axis=1)

            windowFrame = np.concatenate((top_img, bot_img), axis=0)
            windowFrame = cv.resize(windowFrame, (1366, 768))
            loading = False


        #cv.putText(windowFrame, (0, 25),
        #           'First Frame to cut begin: {}'.format(curr_first),
        #           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        cv.imshow('video', windowFrame)
        key = cv.waitKey(0)

        #posição de cima
        if key == ord('w'):
            curr_first = curr_first + 1 if curr_first < size_total \
                                        else curr_first
        elif key == ord('s'):
            curr_first = curr_first - 1 if curr_first < size_total \
                                        else 0

        #posição de baixo
        if key == ord('a'):
            curr_last = curr_last + 1 if curr_last < size_total \
                                      else curr_last
        elif key == ord('d'):
            curr_last = curr_last - 1 if curr_last < size_total \
                                      else 0

        if key == ord(' '):
            open_cut = True

        if key == ord('q'):
            break



