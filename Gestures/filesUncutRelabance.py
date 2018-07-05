dir_name = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/uncut/100x100/P3'
dir_list = [os.path.join(dir_name, l) for l in sorted(os.listdir(dir_name), key=len)]
for idx, file in enumerate(dir_list):
  pos = file.rfind('m')
  print(file)
  os.rename(file, file[:pos + 1] + str(idx) + '.png')

dir_name = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/uncut/complete_depth/P3'
for idx, file in enumerate(dir_list):
    pos = file.rfind('P')
    print(file)
    os.rename(file, file[:pos + 5] + str(idx) + '.xml')
# 1675
# 5314