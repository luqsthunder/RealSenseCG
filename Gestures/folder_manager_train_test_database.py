import os
import numpy as np
import shutil


'''
	@param (db_folder)          Folder which contains all classes.
	@param (test_percent_cut)   Percent to cut and will be in test in test
    @param (test_position_cut)  Which position will cut from total database to use as test tuple e.g: (20, 10)
	@param (suffle)             Randomize both input
	
	throws an error when tuple distance differ from test_percent
	
	example usage:
	copy_from_db_to_train_test(db_folder = './Gestures', 20, (0, 20), shuffle = true)
	copy_from_db_to_train_test(db_folder = './Gestures', 20, (10, 40), shuffle = true)
'''
def copy_from_db_to_train_test(db_folder = '../Gestures/dynamic_poses', test_percent = 10, test_position = (90, 100)):
	for cls_dir in os.listdir(db_folder):
		curr_dir = 
		seqs = [os.path.join(dir, l) for l in os.listdir(dir)]
		seq = np.arange(len(seqs))
		test = seq[len(seq) - int(len(seq) * 1/10):len(seq)]
		seq = np.arange(len(seqs))
		np.random.shuffle(seq)
		test = seq[len(seq) - int(len(seq) * 1/10):len(seq)]

		for (k, t) in enumerate(test):
		  shutil.copytree(seqs[t], '../Gestures/dynamic_poses/F1/test/P2/e' + str(k))
		  shutil.rmtree(seqs[t])
