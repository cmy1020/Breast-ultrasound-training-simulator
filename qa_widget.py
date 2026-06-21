"""乳腺扫查知识问答界面 — 聊天式 Q&A"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QScrollArea, QLabel, QFrame,
)
from PyQt5.QtCore import Qt, QTimer
from knowledge_base import KnowledgeBase


STYLE = """
QAWidget {
    background-color: #2b2b2b;
}
QAWidget > QFrame#title_bar {
    background-color: #1e3a4d;
    border-top: 1px solid #4ec9ff;
}
QAWidget > QScrollArea {
    background-color: #252525;
    border: none;
}
QAWidget QLineEdit {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: #4ec9ff;
}
QAWidget QLineEdit:focus {
    border-color: #4ec9ff;
}
QAWidget QPushButton#btn_send {
    background-color: #1a4a6e;
    color: white;
    border: 1px solid #4ec9ff;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}
QAWidget QPushButton#btn_send:hover {
    background-color: #256b9e;
}
QAWidget QPushButton#btn_send:pressed {
    background-color: #0d3550;
}
"""


def _make_bubble(text: str, role: str) -> str:
    """生成聊天气泡 HTML"""
    if role == "user":
        return (
            f'<div style="text-align:right; margin:6px 0;">'
            f'<span style="display:inline-block; max-width:85%; '
            f'background-color:#1a4a6e; color:#e0e0e0; '
            f'padding:10px 14px; border-radius:12px 12px 2px 12px; '
            f'font-size:13px; text-align:left; word-wrap:break-word;">'
            f'{text}</span></div>'
        )
    else:
        return (
            f'<div style="text-align:left; margin:6px 0;">'
            f'<span style="display:inline-block; max-width:85%; '
            f'background-color:#3a3a3a; color:#e0e0e0; '
            f'padding:10px 14px; border-radius:12px 12px 12px 2px; '
            f'font-size:13px; text-align:left; word-wrap:break-word;">'
            f'<b>💬 </b>{text}</span></div>'
        )


class QAWidget(QWidget):
    """知识问答面板"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.kb = KnowledgeBase()
        self.message_history = []  # [(role, text), ...]

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 标题栏 ──────────────────────────────────────────────
        self.title_bar = QFrame()
        self.title_bar.setObjectName("title_bar")
        self.title_bar.setFixedHeight(28)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(12, 4, 12, 4)
        title_label = QLabel("💬 知识问答")
        title_label.setStyleSheet("color: #4ec9ff; font-size: 12px; font-weight: bold; background: transparent;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addWidget(self.title_bar)

        # ── 对话历史 ────────────────────────────────────────────
        self.chat_label = QLabel()
        self.chat_label.setWordWrap(True)
        self.chat_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.chat_label.setStyleSheet(
            "background-color: #252525; color: #e0e0e0; "
            "padding: 8px; font-size: 13px;"
        )
        self.chat_label.setText(
            '<div style="color:#888; text-align:center; padding:20px;">'
            '输入乳腺扫查相关问题，如："BI-RADS 分级是什么？"</div>'
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_label)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: #252525; }")
        layout.addWidget(self.scroll_area, stretch=1)

        # ── 输入栏 ──────────────────────────────────────────────
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #2b2b2b; border-top: 1px solid #444;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 6, 8, 6)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入问题，按 Enter 发送...")
        self.input_box.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_box, stretch=1)

        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("btn_send")
        self.btn_send.clicked.connect(self._on_send)
        input_layout.addWidget(self.btn_send)

        layout.addWidget(input_frame)

        # 应用样式
        self.setStyleSheet(STYLE)

    def _on_send(self):
        """处理发送事件"""
        text = self.input_box.text().strip()
        if not text:
            return

        self.input_box.clear()

        # 添加用户消息
        self.message_history.append(("user", text))

        # 搜索答案
        result = self.kb.search(text)
        answer = result["answer"]
        self.message_history.append(("bot", answer))

        # 刷新显示
        self._refresh_chat()

    def _refresh_chat(self):
        """刷新对话历史显示"""
        html_parts = []
        for role, text in self.message_history:
            html_parts.append(_make_bubble(text, role))

        full_html = "".join(html_parts)
        self.chat_label.setText(full_html)

        # 自动滚动到底部
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """滚动到对话底部"""
        sb = self.scroll_area.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())
