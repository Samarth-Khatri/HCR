import os
import sys

import cv2
import numpy as np
import qdarkstyle
from PIL import Image
import editdistance
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

import argparse
import json
from typing import Tuple, List

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = './model/line-model/charList.txt'
    fn_summary = './model/line-model/summary.json'
    fn_corpus = './data/corpus.txt'


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


class Handwriting_Recognition(QMainWindow):

    def __init__(self):
        super(Handwriting_Recognition, self).__init__()

        loadUi('Handwriting.ui', self)

        self.decoder_type = DecoderType.BestPath

        self.model = Model(char_list_from_file(), self.decoder_type, must_restore=True)

        self.browseImage_pushButton.clicked.connect(self.BrowseFileDialog)
        self.recogniseText_pushButton.clicked.connect(self.Recognition_Function)
        self.exit_pushButton.clicked.connect(self.Exit_Function)

    @pyqtSlot()
    def BrowseFileDialog(self):
        self.fname, filter = QFileDialog.getOpenFileName(self, 'Open image File', '.\\', "image Files (*.*)")
        if self.fname:
            self.LoadImageFunction(self.fname)
        else:
            print("No Valid File selected.")

    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image)

    def DisplayImage(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()

        self.imglabel.setPixmap(QPixmap.fromImage(outImg))
        self.imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.imglabel.setScaledContents(True)

    @pyqtSlot()
    def Recognition_Function(self):
        self.PreProcessing()

    def PreProcessing(self):
        img = cv2.imread(self.fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        if w > 1000:
            new_w = 1000
            ar = w / h
            new_h = int(new_w / ar)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

        # dilation
        kernel = np.ones((3, 85), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y, w, h)

        img2 = img.copy()

        reco_text = ""

        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (40, 100, 250), 2)
            # cropped_image = img[Y:Y+H, X:X+W]
            roi_line = img_gray[y:y+h, x:x+w]
            # cv2.imshow("ROI", roi_line)
            reco_text = reco_text + self.Inference(roi_line) + "\n"

        self.plainTextEdit.setPlainText(reco_text)
        self.DisplayImage(img2)

    def Inference(self, img):

        # img = cv2.imread(self.fname, cv2.IMREAD_GRAYSCALE)
        # assert img is not None

        preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
        img = preprocessor.process_img(img)

        batch = Batch([img], None, 1)
        recognized, probability = self.model.infer_batch(batch, True)
        print(f'Recognized: "{recognized[0]}"')
        print(f'Probability: {probability[0]}')

        # self.plainTextEdit.setPlainText(recognized[0])

        return recognized[0]

    @pyqtSlot()
    def Exit_Function(self):
        sys.exit()


''' ------------------------ MAIN Function ------------------------- '''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Handwriting_Recognition()
    window.show()
    sys.exit(app.exec_())
