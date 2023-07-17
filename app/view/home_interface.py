# coding:utf-8
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from qfluentwidgets import ScrollArea, isDarkTheme, FluentIcon
from ..common.config import cfg, HELP_URL, REPO_URL, EXAMPLE_URL, FEEDBACK_URL
from ..common.icon import Icon, FluentIconBase
from ..components.link_card import LinkCardView
from ..components.sample_card import SampleCardView
from ..common.style_sheet import StyleSheet


class BannerWidget(QWidget):
    """ Banner widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(336)
        self.vBoxLayout = QVBoxLayout(self)
        self.galleryLabel = QLabel('煤矿皮带跑偏检测系统', self)
        self.banner = QPixmap(':/gallery/images/header1.png')
        self.linkCardView = LinkCardView(self)

        self.galleryLabel.setObjectName('galleryLabel')

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 30, 0, 0)
        self.vBoxLayout.addWidget(self.galleryLabel)
        self.vBoxLayout.addWidget(self.linkCardView, 1, Qt.AlignBottom)
        self.vBoxLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # 标题下方
        self.linkCardView.addCard(
            ':/gallery/images/logo.png',
            self.tr('快速开始'),
            self.tr('本系统适用于煤矿井下皮带跑偏检测情景,专注于煤矿领域智能视觉识别'),
            HELP_URL
        )

        # self.linkCardView.addCard(
        #     FluentIcon.GITHUB,
        #     self.tr('GitHub repo'),
        #     self.tr(
        #         'The latest fluent design controls and styles for your applications.'),
        #     REPO_URL
        # )

        self.linkCardView.addCard(
            FluentIcon.HELP,
            self.tr('获取帮助'),
            self.tr(
                '您可以联系开发人员获取本系统使用建议'),
            EXAMPLE_URL
        )

        self.linkCardView.addCard(
            FluentIcon.FEEDBACK,
            self.tr('提供反馈'),
            self.tr('通过提供反馈帮助开发人员改进系统'),
            FEEDBACK_URL
        )
        

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        w, h = self.width(), 200
        path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        path.addRect(QRectF(0, h-50, 50, 50))
        path.addRect(QRectF(w-50, 0, 50, 50))
        path.addRect(QRectF(w-50, h-50, 50, 50))
        path = path.simplified()

        # draw background color
        if not isDarkTheme():
            painter.fillPath(path, QColor(206, 216, 228))
        else:
            painter.fillPath(path, QColor(0, 0, 0))

        # draw banner image
        pixmap = self.banner.scaled(
            self.size(), transformMode=Qt.SmoothTransformation)
        path.addRect(QRectF(0, h, w, self.height() - h))
        painter.fillPath(path, QBrush(pixmap))


class HomeInterface(ScrollArea):
    """ Home interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.banner = BannerWidget(self)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.__initWidget()
        self.loadSamples()

    def __initWidget(self):
        self.view.setObjectName('view')
        StyleSheet.HOME_INTERFACE.apply(self)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 36)
        self.vBoxLayout.setSpacing(40)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

    def loadSamples(self):
        """ load samples """
        # basic input samples
        basicInputView = SampleCardView(
            self.tr("系统功能"), self.view)
        basicInputView.addSampleCard(
            icon=":/gallery/images/controls/Image.png",
            title="图像识别",
            content=self.tr(
                "从本机中读取图片进行识别"),
            routeKey="imgInterface",
            index=0
        )
        basicInputView.addSampleCard(
            icon=":/gallery/images/controls/MediaPlayerElement.png",
            title="视频识别",
            content=self.tr("从本机中读取视频进行识别"),
            routeKey="videoInterface",
            index=0
        )
        basicInputView.addSampleCard(
            icon=":/gallery/images/controls/WebView.png",
            title="在线识别",
            content=self.tr(
                "从摄像头中读取视频进行识别"),
            routeKey="cameraInterface",
            
            index=0
        )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/DropDownButton.png",
        #     title="DropDownButton",
        #     content=self.tr(
        #         "A button that displays a flyout of choices when clicked."),
        #     routeKey="basicInputInterface",
        #     index=8
        # )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/RadioButton.png",
        #     title="RadioButton",
        #     content=self.tr(
        #         "A control that allows a user to select a single option from a group of options."),
        #     routeKey="basicInputInterface",
        #     index=10
        # )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/Slider.png",
        #     title="Slider",
        #     content=self.tr(
        #         "A control that lets the user select from a range of values by moving a Thumb control along a track."),
        #     routeKey="basicInputInterface",
        #     index=11
        # )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/SplitButton.png",
        #     title="SplitButton",
        #     content=self.tr(
        #         "A two-part button that displays a flyout when its secondary part is clicked."),
        #     routeKey="basicInputInterface",
        #     index=12
        # )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/ToggleSwitch.png",
        #     title="SwitchButton",
        #     content=self.tr(
        #         "A switch that can be toggled between 2 states."),
        #     routeKey="basicInputInterface",
        #     index=14
        # )
        # basicInputView.addSampleCard(
        #     icon=":/gallery/images/controls/ToggleButton.png",
        #     title="ToggleButton",
        #     content=self.tr(
        #         "A button that can be switched between two states like a CheckBox."),
        #     routeKey="basicInputInterface",
        #     index=15
        # )
        self.vBoxLayout.addWidget(basicInputView)

        # date time samples
        dateTimeView = SampleCardView(self.tr('系统正在完善 敬请期待~~~'), self.view)
        # dateTimeView.addSampleCard(
        #     icon=":/gallery/images/controls/TimePicker.png",
        #     title="TimePicker",
        #     content=self.tr(
        #         "A configurable control that lets a user pick a time value."),
        #     routeKey="dateTimeInterface",
        #     index=2
        # )
        self.vBoxLayout.addWidget(dateTimeView)

        # # dialog samples
        # dialogView = SampleCardView(self.tr('Dialog samples'), self.view)
        # dialogView.addSampleCard(
        #     icon=":/gallery/images/controls/Flyout.png",
        #     title="Dialog",
        #     content=self.tr("A frameless message dialog."),
        #     routeKey="dialogInterface",
        #     index=0
        # )
        # dialogView.addSampleCard(
        #     icon=":/gallery/images/controls/ContentDialog.png",
        #     title="MessageBox",
        #     content=self.tr("A message dialog with mask."),
        #     routeKey="dialogInterface",
        #     index=1
        # )
        # dialogView.addSampleCard(
        #     icon=":/gallery/images/controls/ColorPicker.png",
        #     title="ColorDialog",
        #     content=self.tr("A dialog that allows user to select color."),
        #     routeKey="dialogInterface",
        #     index=2
        # )
        # self.vBoxLayout.addWidget(dialogView)

        # # layout samples
        # layoutView = SampleCardView(self.tr('Layout samples'), self.view)
        # layoutView.addSampleCard(
        #     icon=":/gallery/images/controls/Grid.png",
        #     title="FlowLayout",
        #     content=self.tr(
        #         "A layout arranges components in a left-to-right flow, wrapping to the next row when the current row is full."),
        #     routeKey="layoutInterface",
        #     index=0
        # )
        # self.vBoxLayout.addWidget(layoutView)

        # # material samples
        # materialView = SampleCardView(self.tr('Material samples'), self.view)
        # materialView.addSampleCard(
        #     icon=":/gallery/images/controls/Acrylic.png",
        #     title="AcrylicLabel",
        #     content=self.tr(
        #         "A translucent material recommended for panel background."),
        #     routeKey="materialInterface",
        #     index=0
        # )
        # self.vBoxLayout.addWidget(materialView)

        # # menu samples
        # menuView = SampleCardView(self.tr('Menu samples'), self.view)
        # menuView.addSampleCard(
        #     icon=":/gallery/images/controls/MenuFlyout.png",
        #     title="RoundMenu",
        #     content=self.tr(
        #         "Shows a contextual list of simple commands or options."),
        #     routeKey="menuInterface",
        #     index=0
        # )
        # self.vBoxLayout.addWidget(menuView)

        # # scroll samples
        # scrollView = SampleCardView(self.tr('Scrolling samples'), self.view)
        # scrollView.addSampleCard(
        #     icon=":/gallery/images/controls/ScrollViewer.png",
        #     title="ScrollArea",
        #     content=self.tr(
        #         "A container control that lets the user pan and zoom its content smoothly."),
        #     routeKey="scrollInterface",
        #     index=0
        # )
        # self.vBoxLayout.addWidget(scrollView)

        # # state info samples
        # stateInfoView = SampleCardView(self.tr('Status & info samples'), self.view)
        # stateInfoView.addSampleCard(
        #     icon=":/gallery/images/controls/ProgressRing.png",
        #     title="StateToolTip",
        #     content=self.tr(
        #         "Shows the apps progress on a task, or that the app is performing ongoing work that does block user interaction."),
        #     routeKey="statusInfoInterface",
        #     index=0
        # )
        # stateInfoView.addSampleCard(
        #     icon=":/gallery/images/controls/InfoBar.png",
        #     title="InfoBar",
        #     content=self.tr(
        #         "An inline message to display app-wide status change information."),
        #     routeKey="statusInfoInterface",
        #     index=3
        # )
        # stateInfoView.addSampleCard(
        #     icon=":/gallery/images/controls/ProgressBar.png",
        #     title="ProgressBar",
        #     content=self.tr(
        #         "Shows the apps progress on a task, or that the app is performing ongoing work that doesn't block user interaction."),
        #     routeKey="statusInfoInterface",
        #     index=7
        # )
        # stateInfoView.addSampleCard(
        #     icon=":/gallery/images/controls/ProgressRing.png",
        #     title="ProgressRing",
        #     content=self.tr(
        #         "Shows the apps progress on a task, or that the app is performing ongoing work that doesn't block user interaction."),
        #     routeKey="statusInfoInterface",
        #     index=9
        # )
        # stateInfoView.addSampleCard(
        #     icon=":/gallery/images/controls/ToolTip.png",
        #     title="ToolTip",
        #     content=self.tr(
        #         "Displays information for an element in a pop-up window."),
        #     routeKey="statusInfoInterface",
        #     index=1
        # )
        # self.vBoxLayout.addWidget(stateInfoView)

        # # text samples
        # textView = SampleCardView(self.tr('Text samples'), self.view)
        # textView.addSampleCard(
        #     icon=":/gallery/images/controls/TextBox.png",
        #     title="LineEdit",
        #     content=self.tr("A single-line plain text field."),
        #     routeKey="textInterface",
        #     index=0
        # )
        # textView.addSampleCard(
        #     icon=":/gallery/images/controls/NumberBox.png",
        #     title="SpinBox",
        #     content=self.tr(
        #         "A text control used for numeric input and evaluation of algebraic equations."),
        #     routeKey="textInterface",
        #     index=1
        # )
        # textView.addSampleCard(
        #     icon=":/gallery/images/controls/RichEditBox.png",
        #     title="TextEdit",
        #     content=self.tr(
        #         "A rich text editing control that supports formatted text, hyperlinks, and other rich content."),
        #     routeKey="textInterface",
        #     index=6
        # )
        # self.vBoxLayout.addWidget(textView)

        # # view samples
        # collectionView = SampleCardView(self.tr('View samples'), self.view)
        # collectionView.addSampleCard(
        #     icon=":/gallery/images/controls/ListView.png",
        #     title="ListView",
        #     content=self.tr(
        #         "A control that presents a collection of items in a vertical list."),
        #     routeKey="viewInterface",
        #     index=0
        # )
        # collectionView.addSampleCard(
        #     icon=":/gallery/images/controls/DataGrid.png",
        #     title="TableView",
        #     content=self.tr(
        #         "The DataGrid control provides a flexible way to display a collection of data in rows and columns."),
        #     routeKey="viewInterface",
        #     index=1
        # )
        # collectionView.addSampleCard(
        #     icon=":/gallery/images/controls/TreeView.png",
        #     title="TreeView",
        #     content=self.tr(
        #         "The TreeView control is a hierarchical list pattern with expanding and collapsing nodes that contain nested items."),
        #     routeKey="viewInterface",
        #     index=2
        # )
        # self.vBoxLayout.addWidget(collectionView)
