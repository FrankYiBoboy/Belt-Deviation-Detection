import sys
import cv2
import numpy as np
from typing import List

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer,pyqtSlot,QUrl
from PyQt5.QtWidgets import  QWidget,  QFileDialog
from qfluentwidgets import ( FlowLayout, isDarkTheme,StateToolTip,
                             ToolTipFilter, PixmapLabel, ScrollArea, TextEdit,PlainTextEdit,PushButton, PushSettingCard,
                            )

from qfluentwidgets import FluentIcon as FIF
from .gallery_interface import GalleryInterface
from .gallery_interface import GalleryInterface
from ..common.translator import Translator
from app.view.img_process import process, INITIAL_ROI, FURTHER_ROI


class CameraInterface(GalleryInterface):
    """ Date time interface """

    def __init__(self, parent=None):
        t = Translator()
        super().__init__(
            title=t.dateTime,
            subtitle='从摄像头中读取视频进行识别',
            parent=parent
        )
        self.filename = ''
        self.video_thread = None
        self.show()
        # 显示处理词条
        self.stateTooltip = False
        self.inputButton.clicked.connect(self.openImg)

        self.processButton.clicked.connect(self.processStart)
        self.timer = QTimer(self)
        self.processButton.clicked.connect(self.processImage)
        self.timer.timeout.connect(self.processEnd)
        self.clearTextButton.clicked.connect(self.clearText)

    def show(self):

        # 表头按钮
        self.addExampleCard(
            self.tr('读取摄像头视频'),
            self.createWidget(),
            '',
            stretch=1
        )
        # 原始图像显示
        self.srcVideo()
        # 处理图像显示
        self.proImg()

        # 结果显示区域
        self.plainTextEdit = PlainTextEdit(self)
        self.plainTextEdit.setPlainText(
            "此处显示处理结果:"
        )

        # self.textEdit.setMarkdown(
        #     "## 这是一个表题 \n * 这是一个表题 \n * 此处显示 ")
        self.plainTextEdit.setFixedHeight(150)
        self.addExampleCard(
            title=self.tr("结果显示"),
            widget=self.plainTextEdit,
            sourcePath='',
            stretch=1
        )

        # self.inputButton = PushButton(self.tr('点击读取'))
        # 表头按钮
        self.clearTextButton = PushButton(self.tr('清理文本'))
        self.addExampleCard(
            self.tr(''),
            self.clearTextButton,
            '',
            stretch=1
        )

    # 显示原始视频区域
    def srcVideo(self):
        # 显示原始图像区域
        w = ScrollArea()
        self.srcLabel = PixmapLabel(self)
        self.srcLabel.setPixmap(QPixmap(self.filename).scaled(
            775, 1229, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        w.horizontalScrollBar().setValue(0)
        w.setWidget(self.srcLabel)
        w.setFixedSize(775, 430)

        card = self.addExampleCard(
            self.tr('本地视频'),
            w,
            '',
        )
        card.card.installEventFilter(ToolTipFilter(card.card, showDelay=500))
        card.card.setToolTip(self.tr('此处显示摄像头视频'))
        card.card.setToolTipDuration(2000)
    
    # 处理显示图像区域
    def proImg(self):
        # 处理显示图像区域
        w = ScrollArea()
        self.proLabel = PixmapLabel(self)
        self.proLabel.setPixmap(QPixmap(self.filename).scaled(
            775, 1229, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        w.horizontalScrollBar().setValue(0)
        w.setWidget(self.proLabel)
        w.setFixedSize(775, 430)

        card = self.addExampleCard(
            self.tr('检测图像'),
            w,
            '',
        )
        card.card.installEventFilter(ToolTipFilter(card.card, showDelay=500))
        card.card.setToolTip(self.tr('此处显示检测图像'))
        card.card.setToolTipDuration(2000)
    
    # 创建表头按钮
    def createWidget(self, animation=False):
        
        widget = QWidget()
        layout = FlowLayout(widget, animation)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(20)
        layout.setHorizontalSpacing(10)

        self.inputButton = PushButton(self.tr('点击读取'))
        layout.addWidget(self.inputButton)
        self.processButton = PushButton(self.tr('开始处理'))
        layout.addWidget(self.processButton)

        return widget

    # @pyqtSlot()
    # def select_video(self):
    #     self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
    #     if self.video_path:
    #         self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
    #         self.video_player.play()

    # @pyqtSlot()
    # def start_processing(self):
    #     if self.video_path and not self.video_thread:
    #         self.video_thread = VideoThread(self.video_path)
    #         self.video_thread.frame_processed.connect(self.update_image)
    #         self.video_thread.start()
    # 本地文件夹查找图片
    def openImg(self):
        # 打开文件对话框选择图片文件
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            self.filename = filenames[0]

            # image = QPixmap.fromImage(QImage(self.filename))
            self.srcLabel.setPixmap(QPixmap(self.filename).scaled(
            775, 1229, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    # 添加文本结果
    def addText(self,text):
        self.plainTextEdit.appendPlainText(text)

    # 设置状态初始
    def processStart(self):
        self.stateTooltip = StateToolTip('正在进行识别', '请耐心等待~~', self)
        self.stateTooltip.move(510, 30)
        self.stateTooltip.show()    
        self.timer.start(2000)
    # 设置成功
    def processEnd(self):
        self.stateTooltip.setContent('识别成功')
        self.stateTooltip.setState(True)
        self.timer.stop()        

    # 处理图像
    def processImage(self):
        if self.filename is not None:
            time_start = cv2.getTickCount()
            img = cv2.imread(self.filename)
            # img = cv2.imread(self.current_image)
            
            imgCopy = img.copy()
            edges = self.inital_process(img)
            # 初步过滤
            vertices = np.array(
                [[[150, 730], [301, 380], [1453, 380], [1618, 730]]])
            masked = self.region_of_interest(
                edges, vertices=vertices, sign=INITIAL_ROI)
            # 右侧过滤
            # 第一次
            vertices_further_1 = np.array(
                [[[1570, 730], [1500, 540], [1600, 540], [1700, 730]]])
            masked = self.region_of_interest(
                masked, vertices=vertices_further_1, sign=FURTHER_ROI)
            # 第二次
            vertices_further_2 = np.array(
                [[[200, 700], [210, 600], [250, 600], [240, 700]]])
            masked = self.region_of_interest(
                masked, vertices=vertices_further_2, sign=FURTHER_ROI)
            # 右跑偏过滤
            '''
            vertices_further_3 = np.array(
                [[[220, 670], [262, 405], [370, 405], [330, 670]]])
            masked = self.region_of_interest(
                masked, vertices=vertices_further_3, sign=FURTHER_ROI)
            '''
            lines = self.dispose_line(masked)

            # 在plainTextEdit中打印处理结果
            lines_len = f"检测出线条点集对数目: {len(lines)}"
            self.addText(lines_len)
            
            left_line, right_line,re_lines_len,left_len,right_len,re_left_len,re_right_len = self.generate_line(lines)
            
            re_lines_len = f"角度滤波后点集对数目: {re_lines_len}"
            self.addText(re_lines_len)

            left_len = f"左侧线条点集对数目: {left_len}"
            self.addText(left_len)

            right_len = f"右侧线条点集对数目: {right_len}"
            self.addText(right_len)

            re_left_len = f"左侧过滤后线条点集对数目: {re_left_len}"
            self.addText(re_left_len)

            re_right_len = f"左侧过滤后线条点集对数目: {re_right_len}"
            self.addText(re_right_len)
            
            res = self.line_pro_count(imgCopy, left_line, right_line)
            res = f"皮带状况: {res}"
            self.addText(res)

            # 将处理后的图像显示在标签中            
            process_img = QPixmap.fromImage(QImage(imgCopy.data, imgCopy.shape[1], imgCopy.shape[0],
                                                        imgCopy.strides[0], QImage.Format_BGR888))
            
            self.proLabel.setPixmap(process_img.scaled(
            775, 1229, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            time_end = cv2.getTickCount()
            time = (time_end - time_start)/cv2.getTickFrequency()
            time = f"程序时间: {time}S"
            self.addText(time)
            ending_text = "-----------------------------------------------------------------------------------------------"
            self.addText(ending_text)

    def clearText(self):
        self.plainTextEdit.clear()
        self.plainTextEdit.setPlainText(
            "此处显示处理结果:"
        )

