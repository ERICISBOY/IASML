from shiny import App, ui, render, reactive
# Update the import line from
# from shiny.reactive import value, effect
# to:
from shiny.reactive import Value as value
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

# ========== 全局配置 ==========
CSS = """
/* 自定义样式 */
#myFooter {
    background: #f8f9fa;
    padding: 40px 30px;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
    margin-top: auto;
}

.footer-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}

.footer-links {
    display: flex;
    gap: 20px;
}

.custom-download-button {
    color: #fff !important; 
    background-color: #20894d !important;
    border: 1px solid #20894d !important;
    font-size: 16px !important;
    font-weight: bold !important;
    margin: 10px;
    padding: 15px 30px !important;
}

.navbar-brand img {
    height: 70px;
    vertical-align: middle;
    margin-right: 10px;
}

.function-card {
    border-left: 5px solid;
    margin-bottom: 20px;
}

/* 使用视口单位确保精确计算 */
.main-content {
    padding-bottom: 20px !important;
    min-height: auto !important;
}

/* 动态调整策略 */
@media (min-height: 800px) {
    .main-content {
        padding-bottom: 15vh;
    }
}
@media (max-height: 600px) {
    .main-content {
        padding-bottom: 25vh;
    }
}

/* 响应式调整 */
@media (max-width: 768px) {
    .main-content {
        padding-bottom: calc(150px + 4vh) !important;  /* 手机端进一步增加 */
    }
    
    #myFooter {
        flex-direction: column;  /* 改为垂直布局 */
        gap: 10px;
        padding: 20px 15px;
    }
    
    .layout-columns > div:last-child {
        height: 200px !important;  /* 手机端增加占位高度 */
    }
}

/* 新增底部固定样式 */
body {
    display: block;  /* 恢复默认文档流 */
    margin: 0;
}

/* 底部栏优化 */
#myFooter a {
    color: #2d6a4f !important;
    transition: color 0.3s ease;
}

#myFooter a:hover {
    color: #ff8033 !important;
    text-decoration: none;
}

.footer-content h4 {
    font-size: 0.9em;
    margin: 0;
}

.footer-content ul li {
    margin: 8px 0;
}

/* 响应式调整 */
@media (max-width: 992px) {
    .footer-content .shiny-layout-column {
        width: 100% !important;
        border-right: none !important;
        padding: 15px 0 !important;
    }
    
    #myFooter img {
        margin: 0 auto;
        display: block;
    }
}

/* 底部栏最终样式 */
#myFooter {
    min-height: 220px;  /* 最小高度保障 */
    background: #f8f9fa;  /* 浅灰色背景 */
    box-shadow: 0 -4px 20px rgba(0,0,0,0.08);  /* 顶部阴影 */
    padding: 40px 30px;
}

/* 内容容器调整 */
.footer-content {
    max-width: 1400px;
    margin: 0 auto;
}

/* 机构信息区块 */
.institute-info {
    padding-right: 60px;
    border-right: 2px solid #ced4da;
    margin-right: 40px;
    line-height: 1.6;
    max-width: 280px;
}

/* 联系方式列表 */
.contact-links li {
    margin: 12px 0;
    font-size: 0.95em;
}

/* 联系方式美化 */
.contact-links li a {
    display: flex;
    align-items: center;
    gap: 8px;
    transition: transform 0.2s ease;
}

.contact-links li a:hover {
    transform: translateX(5px);
}

/* 响应式优化 */
@media (max-width: 992px) {
    #myFooter {
        padding:40px 5% !important;
    }
    
    .footer-columns {
        flex-direction: column;
        gap:30px !important;
    }
    
    .institute-info, 
    .contact-links,
    #myFooter > div > div:last-child {
        width:100% !important;
        padding:0 !important;
        border-right:none !important;
    }
    
    .institute-info img {
        margin-top:20px !important;
    }
}

@media (max-width: 576px) {
    #myFooter {
        padding:30px 15px !important;
    }
    
    .contact-links li {
        margin:15px 0 !important;
    }
}
"""

# 添加文件路径配置
# 在全局配置部分确认下载路径
DOWNLOAD_PATH = Path(__file__).parent / "downloads"
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)  # 确保目录存在

README_PATH = Path(__file__).parent / "Download_README.md"
AI_PATH = Path(__file__).parent / "Algorithm_Introduction.md"
def create_feature_card(title, items, border_color):
    return ui.div(
        ui.h5(title, style="font-size: 140%; margin-bottom: 20px;"),
        *[ui.p(item, style="font-size: 100%; margin: 5px 0;") for item in items],
        style=f"""
            border-left: 5px solid {border_color};
            padding-left: 15px;
            transition: all 0.3s ease;
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            animation: cardEntrance 0.5s ease;
        """,
        class_="function-card",
        # 添加悬停动画
        **{"data-hover": "true"}  # 需要配合CSS选择器
    )

# ========== 新增分析器实现 ==========
# ========== 修改 AnalysisExecutor 类 ==========
class AnalysisExecutor:
    def __init__(self):
        # 使用指定可写路径作为临时目录
        self.temp_dir = Path("/home/duckfarm/lxzhu/IASML_online_zh")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # 新增上传文件存储目录
        self.upload_dir = self.temp_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

    async def execute(self, cmd: list) -> dict:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            work_dir = self.temp_dir / timestamp
            work_dir.mkdir(exist_ok=True)
            
            # 复制上传文件到工作目录
            for file_type in ["geno", "pheno", "model_params", "model_frame"]:
                if f"--{file_type}" in cmd:
                    idx = cmd.index(f"--{file_type}") + 1
                    src_path = Path(cmd[idx])
                    dest_path = work_dir / src_path.name
                    shutil.copy(src_path, dest_path)
                    cmd[idx] = str(dest_path)  # 更新命令中的文件路径

            # 重构命令构建逻辑
            executable_cmd = ["python", str("/home/duckfarm/lxzhu/IASML_online_zh/IASML.py")]
            
            # 添加基因型文件参数
            if '--tfile' in cmd:
                executable_cmd += ["--tfile", str(Path(cmd[cmd.index('--tfile')+1]).resolve())]
            else:
                executable_cmd += ["--bfile", str(Path(cmd[cmd.index('--bfile')+1]).resolve())]
            
            # 添加必要参数（确保存在性检查）
            required_params = ['--phe', '--phe-pos']
            for param in required_params:
                if param in cmd:
                    idx = cmd.index(param)
                    executable_cmd += [param, str(cmd[idx+1])]
            
            # 模型参数互斥处理（增加存在性检查）
            model_params = ['--model', '--model-params', '--model-frame']
            for param in model_params:
                if param in cmd:
                    idx = cmd.index(param)
                    # 仅对需要文件路径的参数进行解析
                    if param != "--model":  # 预置模型不需要路径解析
                        value = str(Path(cmd[idx+1]).resolve())
                    else:
                        value = cmd[idx+1]  # 直接使用模型名称
                    executable_cmd += [param, value]

            # 替换为异步子进程调用
            process = await asyncio.create_subprocess_exec(
                *executable_cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            # 实时写入日志（保持原有逻辑）
            log_file = work_dir / "execution.log"
            with open(log_file, "w", encoding='utf-8') as f:
                while True:
                    output = await process.stdout.read(512)  # 异步读取输出
                    if not output:
                        break
                    f.write(output.decode('utf-8'))
                    f.flush()

            await process.wait()  # 等待进程完成
            returncode = process.returncode

            # 返回结果时保留目录路径
            return {
                'success': process.returncode == 0,
                'log_path': str(log_file),
                'result_path': str(work_dir),  # 返回完整结果路径
                'temp_dir': str(self.temp_dir)  # 暴露临时目录路径
            }

        except Exception as e:
            logging.error(f"执行失败: {str(e)}")
            return {'success': False, 'error': str(e)}
        # 移除 finally 中的清理代码以保留结果文件
# ========== 修改在线使用界面 ==========
# 在页面头部添加MathJax支持
app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.script("""
            // 心跳检测（每30秒发送一次请求）
            setInterval(() => {
                fetch('/_ping_');
            }, 30000);
        """),
        # ... existing head content...
    ),
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),  # 新增MathJax
        ui.tags.script("""
            MathJax = {
              tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true,
                    packages: {'[+]': ['ams']}
              },
              loader: {load: ['[tex]/ams']},
              startup: {
                ready: () => {
                  MathJax.startup.defaultReady();
                  MathJax.startup.promise.then(() => {
                    console.log('MathJax initialized');
                  });
                }
              }
            };
        """),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"),  # 新增Bootstrap JS
        ui.tags.script("$(function () { $('[data-bs-toggle=\"tooltip\"]').tooltip() })")  
    ),

    # 在线使用页面
    ui.nav_panel("在线使用",
        ui.div(
            ui.layout_columns(
                # 数据上传卡片
                ui.card(
                    ui.card_header("数据上传"),
                    ui.input_radio_buttons(
                        "file_type",
                        "分子标记文件格式:",
                        {
                            "bfile": ui.span(
                                "Plink二进制格式",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="请提供bed、bim、fam三个文件，文件大小不超过100M",
                                    data_bs_toggle="tooltip"
                                )
                            ),
                            "tfile": ui.span(
                                "TXT格式",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="请按照每行一个个体、每列一个分子标记的方式排列",
                                    data_bs_toggle="tooltip"
                                )
                            )
                        }
                    ),
                    ui.input_file("geno_file", "选择分子标记文件", multiple=True),
                    # 修改为行内布局
                    ui.div(
                        ui.div(
                            ui.span(
                                "选择表型文件（TXT格式）",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question", 
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="请将各个性状名放在第一行，各个体名放在第一列。",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                ),
                                style="display: inline-flex; align-items: center;"
                            ),
                            style="margin-bottom: 8px;"
                        ),
                        ui.input_file("pheno_file", "", accept=[".txt"]),
                        style="display: flex; flex-direction: column; gap: 4px;"
                    ),
                    ui.output_ui("pheno_column_selector"),
                    ui.output_ui("covariate_selectors"),
                ),
                # 模型设置卡片
                ui.card(
                    ui.card_header("模型设置"),
                    ui.input_radio_buttons(
                        "model_type", "模型类型",
                        {
                            "prebuilt": ui.span(
                                "预置模型",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="选择特定模型，使用本平台预置的参数搜索策略获取模型最佳参数进行计算",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            ),
                            "params_file": ui.span(
                                "上传超参数文件",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="该选项针对神经网络以外的模型，用户可根据需要自定义模型的超参数",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            ),
                            "keras_model": ui.span(
                                "上传Keras模型",
                                ui.span(
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;"
                                    ),
                                    title="可以直接使用已经训练好的神经网络模型用于预测，请将待预测个体的特征与之前用于训练个体的特征保持一致",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            )
                        }
                    ),
                    # 预置模型选择
                    ui.panel_conditional(
                        "input.model_type == 'prebuilt'",
                        ui.input_select(
                            "model", "选择模型",
                            choices=[
                                "SVM", "Ridge", "Lasso", "Elasticnet",
                                "Decision_tree", "Random_forest", 
                                "LightGBM", "XGBoost", "Linear", "PLS", "GBM","CNN","MLP"
                            ]
                        )
                    ),
                    # 参数文件上传（恢复原始结构）
                    ui.panel_conditional(
                        "input.model_type == 'params_file'",
                        ui.input_file("model_params", "上传超参数文件")
                    ),
                    # Keras模型上传（恢复原始结构）
                    ui.panel_conditional(
                        "input.model_type == 'keras_model'",
                        ui.input_file("model_frame", "上传Keras模型文件")
                    )
                ),  # 闭合模型设置卡片
                # 新增运行结果卡片
                ui.card(
                    ui.card_header("运行控制"),
                    ui.input_action_button(
                        "run_analysis",
                        "开始计算",
                        class_="btn-primary",
                        style="margin-top: 20px; background-color: #20894d;"
                    ),
                    ui.card(
                        ui.card_header("分析结果"),
                        ui.layout_columns(
                            ui.download_button(
                                "download_pred", 
                                "下载预测结果",
                                class_="custom-download-button",
                                style="margin: 5px;"
                            ),
                            ui.download_button(
                                "download_model", 
                                "下载模型参数",
                                class_="custom-download-button",
                                style="margin: 5px;"
                            ),
                            col_widths=(6, 6)
                        ),
                        ui.output_ui("result_log")
                    )
                )
            ),
            style="padding: 20px;"
        )
    ),  # 闭合在线使用nav_panel
    # page_navbar关键字参数（修复缩进和拼写错误）
    title=ui.div(
        ui.span("IASML", style="font-weight: 600;"),
        ui.div(
            ui.a(
                "IASBreeding主页",
                href="https://iasbreeding.cn",
                class_="btn",
                style=(
                    "background-color: #20894d; color: white; "
                    "position: fixed; right: 30px; top: 20px; z-index: 1000;"  # 改为固定定位
                )
            ),
            style="position: static;"  # 移除父容器定位
        ),       
        ui.a(
            "IASML主页",
            href="https://iasbreeding.cn/IASML_zh",
            class_="btn ms-auto",
            style="background-color: #20894d; color: white; margin-right: 30px;"
        ),
        style=("display: flex; align-items: center; width: 100%; "
               "justify-content: space-between; gap: 20px;")  # 新增gap间距
    ),  
    # ... 其他参数不变 ...
)

# ========== 服务器端定义 ==========
def server(input, output, session):
    session.session_settings = {
        "timeout": 720  # 延长至12小时
    }
    analysis_status = value("pending")  # Instead of reactive.Value()
    log_content = value("")
    last_position = value(0)
    pheno_columns = value([])
    analysis_result = value(None)  # 将原在底部的变量提前
    analysis_lock = asyncio.Lock()
    # 新增软件下载处理器
    @output
    @render.download()


    @reactive.Effect
    @reactive.event(input.pheno_file)
    def _handle_pheno_upload():
        """处理表型文件上传"""
        file_info = input.pheno_file()
        if not file_info:
            return
        
        try:
            df = pd.read_csv(file_info[0]['datapath'], sep='\t', nrows=0)
            pheno_columns.set(list(df.columns))
        except Exception as e:
            ui.notification_show(f"文件读取失败: {str(e)}", duration=5, type="error")

    @reactive.Effect
    @reactive.event(input.geno_file)
    def _check_geno_file():
        """基因型文件校验"""
        file_info = input.geno_file()
        if not file_info:
            return
    
        try:
            for file in file_info:
                file_path = Path(file['datapath'])
                size_mb = file_path.stat().st_size / (1024 ** 2)
                if size_mb > 100:
                    ui.notification_show(f"基因型文件 {file['name']} 大小({size_mb:.1f}MB)超过限制", duration=10, type="error")
                    input.geno_file.set(None)  # 清空无效文件
                    return
        except Exception as e:
            ui.notification_show(f"文件校验失败: {str(e)}", duration=10, type="error")

    @output
    @render.ui
    def pheno_column_selector():
        """生成表型列选择器"""
        if not pheno_columns():
            return ui.div("请先上传表型文件", class_="text-muted")
            
        return ui.input_select(
            "selected_pheno",
            "选择表型列：",
            {str(i+1): col for i, col in enumerate(pheno_columns())}
        )

    @output
    @render.ui
    def covariate_selectors():
        """生成协变量选择器"""
        if not pheno_columns():
            return ui.div("请先上传表型文件", class_="text-muted")
            
        return ui.TagList(
            ui.input_selectize(
                "factor_covars",
                "因子型协变量（分类变量）:",
                {col: f"{col} (列号: {i+1})" for i, col in enumerate(pheno_columns())},
                multiple=True,
                options={'placeholder': '选择分类变量...'}
            ),
            ui.div(style="margin-top: 15px;"),  # 改用div添加间距
            ui.input_selectize(
                "numeric_covars",
                "数值型协变量（连续变量）:", 
                {col: f"{col} (列号: {i+1})" for i, col in enumerate(pheno_columns())},
                multiple=True,
                options={'placeholder': '选择连续变量...'}
            )
        )

    # 修改分析方法
    @reactive.Effect
    @reactive.event(input.run_analysis)
    async def _run_analysis():
        analysis_status.set("running")
        log_content.set("")
        last_position.set(0)
        analysis_result.set(None)
        try:
            # 获取用户选择
            pheno_pos = input.selected_pheno()
            factor_covars = input.factor_covars()
            numeric_covars = input.numeric_covars()
            
            # 动态模型参数处理（修复逻辑）
            cmd = []
            if input.model_type() == "prebuilt":
                cmd += ["--model", input.model()]
            elif input.model_type() == "params_file":
                if not input.model_params():
                    raise ValueError("请上传超参数文件")
                cmd += ["--model-params", str(Path(input.model_params()[0]['datapath']))]
            elif input.model_type() == "keras_model":
                if not input.model_frame():
                    raise ValueError("请上传Keras模型文件")
                cmd += ["--model-frame", str(Path(input.model_frame()[0]['datapath']))]
            
            # 基础参数（修复文件路径处理）
            cmd += [
                "python", "IASML.py",
                "--phe", str(Path(input.pheno_file()[0]['datapath'])),
                "--phe-pos", pheno_pos,
                "--f", ",".join(factor_covars) if factor_covars else "0",
                "--n", ",".join(numeric_covars) if numeric_covars else "0"
            ]

            # 基因型文件处理（修复路径获取）
            if input.file_type() == "bfile":
                geno_path = Path(input.geno_file()[0]['datapath'])
                cmd += ["--bfile", str(geno_path.parent / geno_path.stem)]
            else:
                cmd += ["--tfile", str(Path(input.geno_file()[0]['datapath']))]

            # 执行分析
            executor = AnalysisExecutor()
            result = await executor.execute(cmd=cmd)
            analysis_result.set(result)
            
        except Exception as e:
            # 添加错误提示
            ui.notification_show(f"分析失败: {str(e)}", duration=10, type="error")
            logging.exception("分析错误详情：")

    # 确保结果变量已定义
    analysis_result = reactive.Value(None)
    
    # 新增下载处理器（同步版本）
    @output
    @render.download()
    def download_pred():
        """下载当前计算的预测结果"""
        if not analysis_result() or not analysis_result().get('success'):
            raise Exception("请先完成计算")
        
        # 直接使用当前计算目录
        current_work_dir = Path(analysis_result().get('result_path'))
        pred_files = list(current_work_dir.glob("result_predict.txt"))
        if pred_files:
            return str(pred_files[0])
        raise Exception("预测文件未找到")

    @output
    @render.download()
    def download_model():
        """下载当前计算的模型参数"""
        if not analysis_result() or not analysis_result().get('success'):
            raise Exception("请先完成计算")
        
        current_work_dir = Path(analysis_result().get('result_path'))
        model_files = list(current_work_dir.glob("result_model.txt")) + list(current_work_dir.glob("result_model.keras"))
        if model_files:
            return str(model_files[0])
        raise Exception("模型文件未找到")

    # 修正结果显示逻辑（删除重复定义）
    @output
    @render.ui
    @reactive.calc
    def result_log():
        status = analysis_status()  # 假设这是一个响应式变量
        
        if status == "pending":
            return ui.div("等待分析开始...", class_="text-muted")
        elif status == "running":

            log_path = Path(analysis_result().get('result_path')) / "IASML.log"
            try:
                if log_path.exists():
                    import subprocess
                    result = subprocess.run(['tail', '-n', '100', str(log_path)], capture_output=True, text=True)
                    log_content = result.stdout
                    return ui.div(
                        ui.pre(
                            log_content,
                            style="height: 400px; overflow-y: auto; background-color: #f8f9fa; padding: 10px;"
                        ),
                        class_="live-log"
                    )
                else:
                    return ui.div("正在等待日志文件生成...", class_="text-info")
            except Exception as e:
                return ui.div(f"日志读取失败: {str(e)}", class_="text-danger")
        else:
            return ui.div("未知状态！", class_="text-danger")

from fastapi.responses import RedirectResponse  # 新增导入

# ========== 应用实例化 ==========
app = App(app_ui, server)

# ========== 启动配置 ========== 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
