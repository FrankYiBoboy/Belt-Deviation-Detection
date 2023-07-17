# coding: utf-8
from PyQt5.QtCore import QObject


class Translator(QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.text = self.tr('Text')
        # self.view = self.tr('View')
        # self.menus = self.tr('Menus')
        self.icons = self.tr('图像识别')
        # self.layout = self.tr('Layout')
        # self.dialogs = self.tr('Dialogs')
        # self.scroll = self.tr('Scrolling')
        # self.material = self.tr('Material')
        self.dateTime = self.tr('在线识别')
        self.basicInput = self.tr('视频识别')
        # self.statusInfo = self.tr('Status & info')