import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QTextCursor

class MarkdownEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Markdown编辑器")
        self.setGeometry(100, 100, 800, 600)

        self.text_edit = QTextEdit(self)

        self.button = QPushButton("添加三级标题和圆点列表", self)
        self.button.clicked.connect(self.add_heading_and_list)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.markdown_text = "# 标题\n\n**加粗文本**\n\n*斜体文本*\n\n[链接](https://www.example.com)\n\n- 列表项\n- 列表项\n- 列表项"
        self.text_edit.setMarkdown(self.markdown_text)

    def add_heading_and_list(self):
        heading_text = "新的三级标题"
        lines = [1, 2, 3, 4, 5]  # 假设这是线条点集列表

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)

        # 使用 <h3> 标签创建三级标题
        cursor.insertHtml(f"\n<h3>{heading_text}</h3>")

        # 使用 <ul> 标签创建无序列表
        cursor.insertHtml("\n<ul>")

        for line in lines:
            # 将 len(line) 字符串放在 <li> 标签内
            cursor.insertHtml(f"\n<li>{len(line)}</li>")

        cursor.insertHtml("\n</ul>")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MarkdownEditorWindow()
    window.show()
    sys.exit(app.exec_())