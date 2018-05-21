import numpy as np
import cv2 as cv
import os
from shutil import copyfile


def copy_videos_to_dest(first, last, class_number):

    folder_src_nd = '../../Gestures/dynamic_poses/uncut/norm_depth'
    folder_src_cd = '../../Gestures/dynamic_poses/uncut/complete_depth'
    folder_src_md = '../../Gestures/dynamic_poses/uncut/mod'
    folder_src_10 = '../../Gestures/dynamic_poses/uncut/100x100'

    folder_dest_nd = '../../Gestures/dynamic_poses/DB/norm_depth'
    folder_dest_cd = '../../Gestures/dynamic_poses/DB/complete_depth'
    folder_dest_md = '../../Gestures/dynamic_poses/DB/mod'
    folder_dest_10 = '../../Gestures/dynamic_poses/DB/100x100'

    curr_folder_src_nd = os.path.join(folder_src_nd,
                                      'P' + str(class_number + 1))
    curr_folder_src_cd = os.path.join(folder_src_cd,
                                      'P' + str(class_number + 1))
    curr_folder_src_md = os.path.join(folder_src_md,
                                      'P' + str(class_number + 1))
    curr_folder_src_10 = os.path.join(folder_src_10,
                                      'P' + str(class_number + 1))

    videos_count = len(os.listdir(os.path.join(folder_dest_nd,
                                               'P' + str(class_number + 1))))

    curr_folder_dst_nd = os.path.join(folder_dest_nd,
                                      'P' + str(class_number + 1), 'e' +
                                      str(videos_count))

    curr_folder_dst_cd = os.path.join(folder_dest_cd,
                                      'P' + str(class_number + 1), 'e' +
                                      str(videos_count))

    curr_folder_dst_md = os.path.join(folder_dest_md,
                                      'P' + str(class_number + 1), 'e' +
                                      str(videos_count))

    curr_folder_dst_10 = os.path.join(folder_dest_10,
                                      'P' + str(class_number + 1), 'e' +
                                      str(videos_count))

    if not os.path.exists(curr_folder_dst_nd):
        os.makedirs(curr_folder_dst_nd)

    if not os.path.exists(curr_folder_dst_cd):
        os.makedirs(curr_folder_dst_cd)

    if not os.path.exists(curr_folder_dst_md):
        os.makedirs(curr_folder_dst_md)

    if not os.path.exists(curr_folder_dst_10):
        os.makedirs(curr_folder_dst_10)

    cont_img_fps = 0
    for video_frames_idx in range(first, last + 1):
        img_src_name = 'im' + str(video_frames_idx) + '.png'
        img_dst_name = 'im' + str(cont_img_fps) + '.png'

        img_src_name_cd = 'im' + str(video_frames_idx) + '.xml'
        img_dst_name_cd = 'im' + str(cont_img_fps) + '.xml'

        copyfile(os.path.join(curr_folder_src_nd, img_src_name),
                 os.path.join(curr_folder_dst_nd, img_dst_name))

        copyfile(os.path.join(curr_folder_src_cd, img_src_name_cd),
                 os.path.join(curr_folder_dst_cd, img_dst_name_cd))

        copyfile(os.path.join(curr_folder_src_md, img_src_name),
                 os.path.join(curr_folder_dst_md, img_dst_name))

        copyfile(os.path.join(curr_folder_src_10, img_src_name),
                 os.path.join(curr_folder_dst_10, img_dst_name))

        cont_img_fps += 1


folder_src = '../../Gestures/dynamic_poses/uncut/norm_depth'
folder_dest = '../../Gestures/dynamic_poses/DB/norm_depth'
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

for idx, uncut_vid_path in enumerate(sorted(os.listdir(folder_src), key=len)):
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
                    show_top.append(np.zeros((480, 640, 3)))

            for idx_rng in range(curr_last - 4, curr_last + 1):
                if curr_first < idx_rng:
                    show_botton.append(cv.imread(uncut_video_folder[idx_rng]))
                else:
                    show_botton.append(np.zeros((480, 640, 3)))

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

            cv.putText(window_frame,
                       'First Frame to cut is: {}'.format(curr_first), (0, 25),
                       font, font_scale, (0, 255, 0), line_type)
            cv.putText(window_frame,
                       'Last Frame to cut is: {}'.format(curr_last), (0, 50),
                       font, font_scale, (0, 255, 0), line_type)
            cv.putText(window_frame,
                       'amount of images: {}'.format(size_total), (0, 75),
                       font, font_scale, (0, 255, 0), line_type)

            cv.putText(window_frame,
                       'amount cut: {}'.format(len(cut_pieces[idx])), (0, 100),
                       font, font_scale, (0, 255, 0), line_type)

            loading = False

        cv.imshow('video', window_frame)

        cv.putText(curr_video_show[fps], 'fps: {}'.format(fps), (0, 25), font,
                   font_scale, (0, 255, 0), line_type)
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
        elif key == ord(' '):
            cut = True
        elif key == ord('q'):
            break

        if cut:
            cut_pieces[idx].append((curr_first, curr_last))
            copy_videos_to_dest(curr_first, curr_last, idx)
            initial_cut = curr_last
            curr_first = curr_last
            if curr_first == len(uncut_video_folder) - 1:
                break

            curr_last = max(curr_first, min(curr_first + 10,
                                            size_total - 1))
            loading = True
            cut = False

cv.destroyAllWindows()

for idx, gestures_cut_folder in enumerate(cut_pieces):
    file = open('gesture(' + str(idx) + ')_cut_info.txt', 'w')
    for item in gestures_cut_folder:
        file.write("{}, ".format(item))
    file.close()
