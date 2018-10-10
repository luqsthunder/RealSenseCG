from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import time

import os

import cv2 as cv

root = Tk()


class LoadPPMs(Frame):

    def __init__(self):
        super().__init__()
        self.master.title('PPM BIN to ASCII')
        self.master.minsize(350, 150)
        self.grid(sticky=E+W+N+S)

        self.btn = ttk.Button(self, text='Select PPMs to convert',
                              command=self.open_ppms)
        self.btn.grid(row=0, column=0, pady=2, padx=2, sticky=W,
                      columnspan=3)
        self.label_in_dir = Label(self, text='No input dir Selected')
        self.label_in_dir.grid(row=1, column=0, pady=2, padx=2, sticky=W,
                               columnspan=3)

        self.btn_output_dir = ttk.Button(self, text='Select Dest Folder',
                                         command=self.name_out_dir)
        self.btn_output_dir.grid(row=2, column=0, pady=2, padx=2,
                                 sticky=E+W+N+S, columnspan=3)
        self.label_out_dir = Label(self, text='No Output dir Selected')
        self.label_out_dir.grid(row=3, column=0, pady=2, padx=2,
                                sticky=W, columnspan=3)

        self.btn_begin = ttk.Button(self, text='Begin convert',
                                    command=self.begin_convert)
        self.btn_begin.grid(row=4, column=0, pady=2, padx=2, sticky=W,
                            columnspan=3)

        self.prog_bar = ttk.Progressbar(self, orient='horizontal',
                                        length=100, mode='determinate')
        self.prog_bar.grid(row=5, column=0, pady=2, padx=2, sticky=W+E)

        self.names = []
        self.dest = ''

        self.prog_bar['value'] = 0
        self.prog_bar['maximum'] = 100

    def open_ppms(self):
        self.names = filedialog.askopenfilenames(title='Select PPMs to convert')
        cut_pos = self.names[0].rfind('frame') - 1
        self.label_in_dir['text'] = self.names[0][:cut_pos]

    def name_out_dir(self):
        self.dest = filedialog.askdirectory(title='Select PPMs dest dir')
        self.label_out_dir['text'] = 'Dest Folder ->:' + self.dest

    def begin_convert(self):
        self.prog_bar['maximum'] = len(self.names)
        for k, it in enumerate(self.names):
            new_name = it[it.rfind('frame'): it.rfind('.ppm')] + '_ascii.ppm'
            self.convert_to_ppm(it, os.path.join(self.dest, new_name))
            self.prog_bar.step(1)
            self.update_idletasks()
        self.prog_bar['progress'] = len(self.names)

    def convert_to_ppm(self, dir_im, name):
        im = cv.imread(dir_im, cv.CV_16UC1)
        file = open(name, 'a+')
        file.write('P3\n')
        file.write('{} {}\n'.format(im.shape[1], im.shape[0]))
        file.write('65535\n')
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                val = im[y, x]
                if x < im.shape[1] - 1:
                    file.write('{} {} {}  '.format(val, val, val))
                else:
                    file.write('{} {} {}'.format(val, val, val))
                self.update_idletasks()
            file.write('\n')
        file.close()


app = LoadPPMs()
app.mainloop()


