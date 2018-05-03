import numpy as np
import cv2 as cv
import os


folder_src = '../../Gestures/dynamic_poses/uncut/norm_depth'
folder_dest = '../../Gestures/dynamic_poses/DB'
folder_trash_dest = '../../Gestures/dynamic_poses/trash'

cv.namedWindow('video')
cv.namedWindow('currentCut')

curr_first = 0
length_show = 10
curr_last = curr_first + length_show

# video information
current_gesture = 0  # PX
current_video = 0    # EX
current_video_frame_cont = length_show

# images to show in cut window
show_top = []
show_botton = []

cut_pieces = []
trash_cut_pieces = []
curr_video_show = []

initial_cut = 0

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
line_type = 2
fps = 0
range_video = curr_last - curr_first
window_frame = np.array((10, 10))

for idx, uncut_vid_path in enumerate(os.listdir(folder_src)):
    cut_pieces.append([])
    uncut_video_folder = sorted(os.listdir(os.path.join(folder_src,
                                                        uncut_vid_path)),
                                key=len)
    uncut_video_folder = [os.path.join(folder_src, uncut_vid_path, l)
                          for l in uncut_video_folder]
    size_total = len(uncut_video_folder)
    if size_total == 0:
        continue

    cutting = True
    cut = False
    open_cut = False

    show_top.clear()
    show_botton.clear()

    loading = True
    while cutting:

        if loading:
            show_top.clear()
            show_botton.clear()

            range_video = curr_last - curr_first
            fps = 0
            curr_video_show.clear()
            for idx_rng in range(curr_first, curr_last):
                curr_video_show.append(cv.imread(uncut_video_folder[idx_rng]))

            for idx_rng in range(curr_first, curr_first + 5):
                if idx_rng < curr_last:
                    show_top.append(cv.imread(uncut_video_folder[idx_rng]))
                else:
                    show_top.append(np.zeros((640, 480, 3)))

            for idx_rng in range(curr_last - 5, curr_last):
                if curr_first < idx_rng:
                    show_botton.append(cv.imread(uncut_video_folder[idx_rng]))
                else:
                    show_botton.append(np.zeros((640, 480, 3)))

            top_img = show_top[0]
            for idx_top_img in range(1, len(show_top)):
                top_img = np.concatenate((top_img, show_top[idx_top_img]),
                                         axis=1)

            bot_img = show_botton[0]
            for idx_top_img in range(1, len(show_top)):
                bot_img = np.concatenate((bot_img, show_botton[idx_top_img]),
                                         axis=1)

            window_frame = np.concatenate((top_img, bot_img), axis=0)
            window_frame = cv.resize(window_frame, (1366, 768))
            loading = False

        cv.putText(window_frame,
                   'First Frame to cut is: {}'.format(curr_first), (0, 25),
                   font, font_scale, (0, 255, 0), line_type)
        cv.putText(window_frame,
                   'Last Frame to cut is: {}'.format(curr_last), (0, 50),
                   font, font_scale, (0, 255, 0), line_type)
        cv.putText(window_frame,
                   'amount of images: {}'.format(size_total), (0, 75),
                   font, font_scale, (0, 255, 0), line_type)

        cv.imshow('video', window_frame)

        cv.imshow('currentCut', curr_video_show[fps])
        fps = (fps + 1) % range_video

        key = cv.waitKey(60)

#       posição de cima
        if key == ord('w'):
            curr_first = max(initial_cut, min(curr_first + 1, curr_last))
            loading = True
        elif key == ord('s'):
            curr_first = max(initial_cut, min(curr_first - 1, curr_last))
            loading = True

#       posição de baixo
        if key == ord('d'):
            curr_last = max(curr_first + 1, min(curr_last + 1, size_total - 1))
            loading = True
        elif key == ord('a'):
            curr_last = max(curr_first + 1, min(curr_last - 1, size_total - 1))
            loading = True

        if key == ord('q'):
            break

        if cut:
            cut_pieces[idx].append((curr_first, curr_last))
            curr_last = initial_cut
            loading = False
            cut = False

cv.destroyAllWindows()


