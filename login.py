import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTranslator
from qframelesswindow import FramelessWindow, StandardTitleBar
from qfluentwidgets import setThemeColor, FluentTranslator, InfoBar, InfoBarIcon, InfoBarPosition, FluentIcon, MessageBox
from PyQt5.QtGui import QIcon

from app.common.config import cfg
from app.view.main_window import MainWindow
from Ui_LoginWindow import Ui_Login


class loginWindow(FramelessWindow, Ui_Login):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 更改主题色
        setThemeColor("#28afe9")

        # 设置标题栏
        self.setTitleBar(StandardTitleBar(self))
        self.titleBar.raise_()

        # 设置窗口图标
        # self.label.setScaledContents(False)
        self.setWindowIcon(QIcon(":images/logo.png"))
        self.setWindowTitle("皮带跑偏检测系统登录界面")
        self.resize(1000, 650)

        # 窗口居中
        rect = QApplication.desktop().availableGeometry()
        w, h = rect.width(), rect.height()
        self.move(w//2-self.width()//2, h//2-self.height()//2)
        self.setStyleSheet("LoginWindow{background: rgba(242, 242, 242, 0.8)}")

        # 调整样式
        self.titleBar.titleLabel.setStyleSheet("""
            QLabel{
                background: transparent;
                font: 14px 'Segoe UI';
                padding: 0 5px;
                color: white
            }
        """)
        # 登录界面
        self.fondButton.clicked.connect(self.createFondInfoBar)
        self.regButton.clicked.connect(self.createRegistInfoBar)
        self.loginButton.clicked.connect(self.loginClicked)

    # 设置找回密码弹窗
    def createFondInfoBar(self):
        content = "密码丢失请联系开发者或数据库管理员,暂不支持在线修改。"
        fond = InfoBar(
            icon=InfoBarIcon.INFORMATION,
            title='警告',
            content=content,
            orient=Qt.Vertical,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self
        )
        fond.show()

    # 注册账号弹窗
    def createRegistInfoBar(self):
        content = "本软件为局域网软件,如需添加账户,请联系数据库管理员更新用户数据库。"
        regist = InfoBar.new(
            icon=FluentIcon.MESSAGE,
            title='温馨提示',
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM,
            duration=2000,
            parent=self
        )
        regist.setCustomBackgroundColor('white', '#202020')

    # 登录按钮slot
    def loginClicked(self):
        user = self.userEdit.text()
        password = self.passwordEdit.text()
        userContent = "admin"
        passwordContent = "123456"

        # 用户名密码正确关闭登录界面切换到主页
        if user == userContent and password == passwordContent:
            showWindow = MainWindow()
            showWindow.show()
            self.close()
        else:
            self.defeatShowDialog()

    # 用户名或密码错误弹窗
    def defeatShowDialog(self):
        title = "您输入的用户名或密码不正确:"
        content = "如果您的密码丢失或遗忘,请联系开发者或数据库管理员重置密码。"
        defeat = MessageBox(title, content, self)
        defeat.exec()


if __name__ == '__main__':
    

    if cfg.get(cfg.dpiScale) == "Auto":
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    else:
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # create application
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    # 添加翻译 面向国际化
    locale = cfg.get(cfg.language).value
    translator = FluentTranslator(locale)
    galleryTranslator = QTranslator()
    galleryTranslator.load(locale, "gallery", ".", ":/gallery/i18n")

    app.installTranslator(translator)
    app.installTranslator(galleryTranslator)
    # 显示界面
    w = loginWindow()
    # showWindow = MainWindow()
    w.show()
    app.exec_()
