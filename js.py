from shiny import App, ui
from pathlib import Path
import os
from shiny import App, ui, render, reactive
from shiny.reactive import value, effect
from pathlib import Path
import hashlib
from shiny.ui import nav_control
import os
import asyncio
import tempfile
import shutil
import json
from typing import Optional
import time
import zipfile
import pandas as pd
import logging
import nest_asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from queue import Empty

def app_ui(request):
    return ui.tags.html(
        ui.tags.head(
            ui.tags.meta(charset="UTF-8")
        ),
        ui.tags.body(
            ui.HTML(open("index.html", encoding="utf-8").read())
          )
    )

# 保留原有服务器逻辑
def server(input, output, session):
    pass

# 配置静态资源（保持原有设置）
current_dir = Path(__file__).parent.resolve()  # 添加resolve()转换为绝对路径

# 修正 App 配置（修复静态资源路径）
app = App(
    app_ui,
    server,
    static_assets={
        "/": str(current_dir)  # 转换为字符串格式
    }
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)