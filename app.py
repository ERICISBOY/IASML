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

# ========= 全局路径配置（本地 + 云端兼容） =========
# 本地开发时，此路径一般不存在，会自动退回到当前文件所在目录。
# 在阿里云服务器上，你的实际工作目录是 /home/duckfarm/lxzhu/IASML_online_zh。
DEFAULT_WORK_ROOT = Path("/home/duckfarm/lxzhu/IASML_online_zh")
if DEFAULT_WORK_ROOT.exists():
    WORK_ROOT = DEFAULT_WORK_ROOT
else:
    WORK_ROOT = Path(__file__).parent

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

# 预置模型名称映射：界面显示名 -> IASML 命令行模型名
MODEL_NAME_MAP = {
    "SVM": "svm",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "Elasticnet": "elasticnet",
    "Decision_tree": "decision_tree",
    "Random_forest": "random_forest",
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "Linear": "linear",
    "PLS": "pls",
    "GBM": "gbm",
    "CNN": "cnn",
    "MLP": "mlp",
}
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
        # 使用统一的工作根目录（本地为项目目录，云端为 /home/duckfarm/lxzhu/IASML_online_zh）
        self.temp_dir = WORK_ROOT
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # 新增上传文件存储目录
        self.upload_dir = self.temp_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

    async def execute(self, cmd: list) -> dict:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            work_dir = self.temp_dir / timestamp
            work_dir.mkdir(exist_ok=True)

            # 重构命令构建逻辑，始终调用 WORK_ROOT 下的 IASML.py
            executable_cmd = ["python", str(WORK_ROOT / "IASML.py")]
            
            # 添加基因型文件参数
            if '--tfile' in cmd:
                executable_cmd += ["--tfile", str(Path(cmd[cmd.index('--tfile')+1]).resolve())]
            else:
                executable_cmd += ["--bfile", str(Path(cmd[cmd.index('--bfile')+1]).resolve())]
            
            # 添加必要参数（确保存在性检查），包括统一的输出前缀 --out
            required_params = ['--phe', '--phe-pos', '--out']
            for param in required_params:
                if param in cmd:
                    idx = cmd.index(param)
                    executable_cmd += [param, str(cmd[idx+1])]
            
            # 模型 / 参数文件 / Keras 模型
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

            # gather 集成参数（可能跟多个模型名一起）
            if '--gather' in cmd:
                g_idx = cmd.index('--gather')
                gather_args = ['--gather']
                # 收集紧随其后的所有非选项 token 作为模型名
                i = g_idx + 1
                while i < len(cmd) and not str(cmd[i]).startswith('--'):
                    gather_args.append(str(cmd[i]))
                    i += 1
                if len(gather_args) > 1:
                    executable_cmd += gather_args

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
        ui.tags.script("$(function () { $('[data-bs-toggle=\"tooltip\"]').tooltip() })"),
        # 简单语言切换脚本：遍历 .lang-en / .lang-zh 控制显示（每次默认英文）
        ui.tags.script("""
            document.addEventListener('DOMContentLoaded', function () {
                function setLang(lang) {
                    var enEls = document.querySelectorAll('.lang-en');
                    var zhEls = document.querySelectorAll('.lang-zh');
                    if (lang === 'zh') {
                        zhEls.forEach(function (el) { el.style.display = ''; });
                        enEls.forEach(function (el) { el.style.display = 'none'; });
                    } else {
                        enEls.forEach(function (el) { el.style.display = ''; });
                        zhEls.forEach(function (el) { el.style.display = 'none'; });
                    }
                }
                // 每次加载页面都先强制英文
                setLang('en');
                var enLink = document.getElementById('lang-switch-en');
                var zhLink = document.getElementById('lang-switch-zh');
                if (enLink) {
                    enLink.addEventListener('click', function (evt) {
                        evt.preventDefault();
                        setLang('en');
                    });
                }
                if (zhLink) {
                    zhLink.addEventListener('click', function (evt) {
                        evt.preventDefault();
                        setLang('zh');
                    });
                }
            });
        """),
    ),

    # 在线使用页面
    ui.nav_panel(
        ui.div(
            ui.span("在线使用", class_="lang-zh"),
            ui.span("Online usage", class_="lang-en"),
        ),
        ui.div(
            ui.layout_columns(
                # 数据上传卡片
                ui.card(
                    ui.card_header(
                        ui.span("数据上传", class_="lang-zh"),
                        ui.span("Data upload", class_="lang-en"),
                    ),
                    ui.input_radio_buttons(
                        "file_type",
                        ui.div(
                            ui.span("分子标记文件格式:", class_="lang-zh"),
                            ui.span("Molecular marker file format:", class_="lang-en"),
                        ),
                        {
                            "bfile": ui.span(
                                ui.span("Plink二进制格式", class_="lang-zh"),
                                ui.span("Plink binary format", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question",
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="Please provide bed/bim/fam files, each no larger than 100 MB / 请提供bed、bim、fam三个文件，文件大小不超过100M",
                                    data_bs_toggle="tooltip"
                                )
                            ),
                            "tfile": ui.span(
                                ui.span("TXT格式", class_="lang-zh"),
                                ui.span("TXT format", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question",
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="Each row is one individual and each column is one marker / 请按照每行一个个体、每列一个分子标记的方式排列",
                                    data_bs_toggle="tooltip"
                                )
                            )
                        }
                    ),
                    ui.input_file(
                        "geno_file",
                        ui.div(
                            ui.span("选择分子标记文件", class_="lang-zh"),
                            ui.span("Choose genotype file", class_="lang-en"),
                        ),
                        multiple=True,
                    ),
                    # 修改为行内布局
                    ui.div(
                        ui.div(
                            ui.span(
                                ui.span("选择表型文件（TXT格式）", class_="lang-zh"),
                                ui.span("Choose phenotype file (TXT)", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question", 
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="Put trait names in the first row and individual IDs in the first column / 请将各个性状名放在第一行，各个体名放在第一列。",
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
                    ui.card_header(
                        ui.span("模型设置", class_="lang-zh"),
                        ui.span("Model settings", class_="lang-en"),
                    ),
                    ui.input_radio_buttons(
                        "model_type",
                        ui.div(
                            ui.span("模型类型", class_="lang-zh"),
                            ui.span("Model type", class_="lang-en"),
                        ),
                        {
                            "prebuilt": ui.span(
                                ui.span("预置模型", class_="lang-zh"),
                                ui.span("Predefined model", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question",
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="Choose a specific model and use the platform's built-in search strategy to find the best hyperparameters / 选择特定模型，使用本平台预置的参数搜索策略获取模型最佳参数进行计算",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            ),
                            "params_file": ui.span(
                                ui.span("上传超参数文件", class_="lang-zh"),
                                ui.span("Upload hyperparameter file", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question",
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="For non-neural-network models, you can customize hyperparameters by uploading a config file / 该选项针对神经网络以外的模型，用户可根据需要自定义模型的超参数",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            ),
                            "keras_model": ui.span(
                                ui.span("上传Keras模型", class_="lang-zh"),
                                ui.span("Upload Keras model", class_="lang-en"),
                                ui.tags.i(
                                    class_="fa-solid fa-circle-question",
                                    style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                    title="Use an already trained neural network for prediction; feature layout must match the training data / 可以直接使用已经训练好的神经网络模型用于预测，请将待预测个体的特征与之前用于训练个体的特征保持一致",
                                    data_bs_toggle="tooltip",
                                    data_bs_placement="top"
                                )
                            )
                        }
                    ),
                    # 预置模型 / gather 选择
                    ui.panel_conditional(
                        "input.model_type == 'prebuilt'",
                        ui.TagList(
                            ui.input_select(
                                "model",
                                ui.div(
                                    ui.span("选择模型", class_="lang-zh"),
                                    ui.span("Choose model", class_="lang-en"),
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                        title="Choose one predefined model for training and prediction / 选择一个预置模型用于训练和预测",
                                        data_bs_toggle="tooltip",
                                        data_bs_placement="top"
                                    ),
                                ),
                                choices=[
                                    "SVM", "Ridge", "Lasso", "Elasticnet",
                                    "Decision_tree", "Random_forest", 
                                    "LightGBM", "XGBoost", "Linear", "PLS", "GBM","CNN","MLP"
                                ]
                            ),
                            ui.div(style="margin-top: 10px;"),
                            ui.input_checkbox(
                                "use_gather",
                                ui.div(
                                    ui.span("启用集成策略（gather，在多模型中自动择优）", class_="lang-zh"),
                                    ui.span("Enable ensemble strategy (gather, automatically selects the best model among multiple models)", class_="lang-en"),
                                    ui.tags.i(
                                        class_="fa-solid fa-circle-question",
                                        style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                        title="Turn on automatic model ensemble search; IASML will evaluate multiple base models and choose the best / 启用模型集成自动搜索，IASML 会在多个基础模型中评估并择优",
                                        data_bs_toggle="tooltip",
                                        data_bs_placement="top"
                                    ),
                                ),
                                value=False
                            ),
                            ui.panel_conditional(
                                "input.use_gather",
                                ui.input_selectize(
                                    "gather_models",
                                    ui.div(
                                        ui.span("参与集成的基础模型（留空或选择 ALL 表示全部支持的模型）：", class_="lang-zh"),
                                        ui.span("Base models for ensemble (leave empty or choose ALL for all supported models):", class_="lang-en"),
                                    ),
                                    choices={
                                        "all": "ALL（所有支持的经典模型，推荐）",
                                        "SVM": "SVM",
                                        "Ridge": "Ridge",
                                        "Lasso": "Lasso",
                                        "Elasticnet": "Elasticnet",
                                        "Decision_tree": "Decision_tree",
                                        "Random_forest": "Random_forest",
                                        "LightGBM": "LightGBM",
                                        "XGBoost": "XGBoost",
                                        "Linear": "Linear",
                                        "PLS": "PLS",
                                        "GBM": "GBM",
                                    },
                                    multiple=True,
                                    options={"placeholder": "Default ALL if empty / 不选择则默认 ALL"}
                                )
                            )
                        )
                    ),
                    # 参数文件上传
                    ui.panel_conditional(
                        "input.model_type == 'params_file'",
                        ui.input_file(
                            "model_params",
                            ui.div(
                                ui.span("上传超参数文件", class_="lang-zh"),
                                ui.span("Upload hyperparameter file", class_="lang-en"),
                            ),
                        )
                    ),
                    # Keras模型上传
                    ui.panel_conditional(
                        "input.model_type == 'keras_model'",
                        ui.input_file(
                            "model_frame",
                            ui.div(
                                ui.span("上传Keras模型文件", class_="lang-zh"),
                                ui.span("Upload Keras model file", class_="lang-en"),
                            ),
                        )
                    ),
                    ui.hr(),
                    # DR 选择（多选）
                    ui.input_selectize(
                        "dr_methods",
                        ui.div(
                            ui.span("个体相似度降维 DR（可选，仅支持 SVM/Ridge/Lasso/Elasticnet/Random_forest/Linear/PLS/GBM）:", class_="lang-zh"),
                            ui.span("Individual similarity dimension reduction DR (optional, only supports SVM/Ridge/Lasso/Elasticnet/Random_forest/Linear/PLS/GBM):", class_="lang-en"),
                            ui.tags.i(
                                class_="fa-solid fa-circle-question",
                                style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                title="Use similarity-based dimension reduction between individuals; leave empty to disable, or choose auto / 使用基于个体相似度的降维方法，可留空表示不使用，或选择 auto 自动搜索",
                                data_bs_toggle="tooltip",
                                data_bs_placement="top"
                            ),
                        ),
                        choices={
                            "auto": "auto (automatically selects best DR / 自动选择最佳 DR，推荐)",
                            "euclidean": "euclidean",
                            "cosine": "cosine",
                            "hamming": "hamming",
                            "manhattan": "manhattan",
                            "pearson": "pearson",
                            "van_raden": "van_raden",
                            "yang_grm": "yang_grm",
                            "kl_divergence": "kl_divergence",
                        },
                        multiple=True,
                        options={"placeholder": "No DR if empty / 不选择则不使用 DR"}
                    ),
                    ui.hr(),
                    ui.input_checkbox(
                        "use_val",
                        ui.div(
                            ui.span("上传独立验证集（Val）以计算 Pearson（列号与训练表型相同）", class_="lang-zh"),
                            ui.span("Upload an independent validation set (Val) to calculate Pearson (column index consistent with training phenotype)", class_="lang-en"),
                            ui.tags.i(
                                class_="fa-solid fa-circle-question",
                                style="color: #666; cursor: help; margin-left: 8px; vertical-align: middle;",
                                title="Optional: provide an external validation phenotype file to compute Pearson on an independent set / 可选：上传外部验证表型文件，在独立验证集中计算 Pearson 相关",
                                data_bs_toggle="tooltip",
                                data_bs_placement="top"
                            ),
                        ),
                        value=False
                    ),
                    ui.panel_conditional(
                        "input.use_val",
                        ui.input_file(
                            "val_file",
                            ui.div(
                                ui.span("选择验证集表型文件（TXT，可选）", class_="lang-zh"),
                                ui.span("Choose validation phenotype file (TXT, optional)", class_="lang-en"),
                            ),
                            accept=[".txt"],
                        )
                    )
                ),  # 闭合模型设置卡片
                # 新增运行结果卡片
                ui.card(
                    ui.card_header(
                        ui.span("运行控制", class_="lang-zh"),
                        ui.span("Run control", class_="lang-en"),
                    ),
                    ui.input_action_button(
                        "run_analysis",
                        ui.div(
                            ui.span("开始计算", class_="lang-zh"),
                            ui.span("Start analysis", class_="lang-en"),
                        ),
                        class_="btn-primary",
                        style="margin-top: 20px; background-color: #20894d;"
                    ),
                    ui.card(
                        ui.card_header(
                            ui.span("分析结果", class_="lang-zh"),
                            ui.span("Analysis results", class_="lang-en"),
                        ),
                        ui.layout_columns(
                            ui.download_button(
                                "download_pred", 
                                ui.div(
                                    ui.span("下载预测结果", class_="lang-zh"),
                                    ui.span("Download prediction", class_="lang-en"),
                                ),
                                class_="custom-download-button",
                                style="margin: 5px;"
                            ),
                            ui.download_button(
                                "download_model", 
                                ui.div(
                                    ui.span("下载模型参数", class_="lang-zh"),
                                    ui.span("Download model parameters", class_="lang-en"),
                                ),
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
    # page_navbar关键字参数（标题、语言切换与右上角主页按钮）
    title=ui.div(
        ui.span("IASML", style="font-weight: 600;"),
        ui.div(
            ui.a(
                "EN",
                href="#",
                id="lang-switch-en",
                style="margin-right: 8px; font-size: 0.9rem;",
            ),
            ui.a(
                "中文",
                href="#",
                id="lang-switch-zh",
                style="margin-right: 16px; font-size: 0.9rem;",
            ),
            ui.a(
                "IASBreeding home",
                href="https://iasbreeding.cn",
                class_="btn btn-sm",
                style="background-color: #20894d; color: white;",
            ),
            style="margin-left: auto; display: flex; align-items: center; gap: 4px;",
        ),
        style=("display: flex; align-items: center; width: 100%; "
               "gap: 20px;"),
    ),
    # ... 其他参数不变 ...
)

# ========== 服务器端定义 ==========
def server(input, output, session):
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
            ui.notification_show(f"Failed to read phenotype file / 文件读取失败: {str(e)}", duration=5, type="error")

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
                    ui.notification_show(f"Genotype file {file['name']} size ({size_mb:.1f}MB) exceeds limit / 基因型文件大小超过限制", duration=10, type="error")
                    input.geno_file.set(None)  # 清空无效文件
                    return
        except Exception as e:
            ui.notification_show(f"File validation failed / 文件校验失败: {str(e)}", duration=10, type="error")

    @output
    @render.ui
    def pheno_column_selector():
        """生成表型列选择器"""
        if not pheno_columns():
            return ui.div("Please upload phenotype file first / 请先上传表型文件", class_="text-muted")
            
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
            return ui.div("Please upload phenotype file first / 请先上传表型文件", class_="text-muted")
            
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

            # 基本必填校验，避免传入不合法命令给 IASML
            if not pheno_pos:
                ui.notification_show("Please select a phenotype column / 请选择表型列。", duration=5, type="error")
                analysis_status.set("pending")
                return

            model_type_ui = input.model_type()
            if not model_type_ui:
                ui.notification_show("Please select a model type (predefined / parameter file / Keras) / 请选择模型类型（预置模型 / 超参数文件 / Keras 模型）。", duration=5, type="error")
                analysis_status.set("pending")
                return
            
            # 动态模型参数处理（只构建 IASML 的参数列表，具体 python 调用在 AnalysisExecutor 中构建）
            cmd = []
            if model_type_ui == "prebuilt":
                # 将界面上的模型名称映射为 IASML 命令行模型名（如 Ridge -> ridge）
                model_ui = input.model()
                if not model_ui:
                    ui.notification_show("Please select a specific predefined model / 请选择具体的预置模型。", duration=5, type="error")
                    analysis_status.set("pending")
                    return
                model_cli = MODEL_NAME_MAP.get(model_ui)
                if not model_cli:
                    raise ValueError(f"Unsupported predefined model: {model_ui}")

                use_gather = bool(input.use_gather())
                if use_gather:
                    gather_models_ui = input.gather_models() or []
                    if (not gather_models_ui) or ("all" in gather_models_ui):
                        # 默认使用全部支持 gather 的经典模型
                        cmd += ["--gather", "all"]
                    else:
                        gather_cli = []
                        for m in gather_models_ui:
                            if m == "all":
                                continue
                            mc = MODEL_NAME_MAP.get(m)
                            if mc is None:
                                raise ValueError(f"gather 中包含不支持的模型: {m}")
                            gather_cli.append(mc)
                        if not gather_cli:
                            cmd += ["--gather", "all"]
                        else:
                            cmd += ["--gather", *gather_cli]
                else:
                    cmd += ["--model", model_cli]
            elif model_type_ui == "params_file":
                if not input.model_params():
                    raise ValueError("Please upload a hyperparameter file / 请上传超参数文件")
                cmd += ["--model-params", str(Path(input.model_params()[0]['datapath']))]
            elif model_type_ui == "keras_model":
                if not input.model_frame():
                    raise ValueError("Please upload a Keras model file / 请上传Keras模型文件")
                cmd += ["--model-frame", str(Path(input.model_frame()[0]['datapath']))]
            
            # 基础参数（表型、协变量、统一输出前缀）
            cmd += [
                "--phe", str(Path(input.pheno_file()[0]['datapath'])),
                "--phe-pos", pheno_pos,
                "--f", ",".join(factor_covars) if factor_covars else "0",
                "--n", ",".join(numeric_covars) if numeric_covars else "0",
                "--out", "result",  # 固定输出前缀，IASML 将生成 result_predict.txt / result_model.txt
            ]
            
            # 基因型文件处理
            if input.file_type() == "bfile":
                # 三件套：从上传列表中根据扩展名识别 bed/bim/fam
                files = input.geno_file()
                bed_file = next((f for f in files if f['name'].lower().endswith('.bed')), None)
                bim_file = next((f for f in files if f['name'].lower().endswith('.bim')), None)
                fam_file = next((f for f in files if f['name'].lower().endswith('.fam')), None)
                if not (bed_file and bim_file and fam_file):
                    raise ValueError("Please upload Plink-format .bed/.bim/.fam files / 请上传Plink格式的 .bed/.bim/.fam 三个文件")

                # 为本次运行创建一个工作目录，放在统一的 WORK_ROOT 下
                geno_root = WORK_ROOT / "plink_uploads"
                geno_root.mkdir(parents=True, exist_ok=True)
                run_dir = geno_root / time.strftime("%Y%m%d-%H%M%S")
                run_dir.mkdir(exist_ok=True)

                # 使用原始文件名复制到 run_dir
                bed_path = run_dir / Path(bed_file['name']).name
                bim_path = run_dir / Path(bim_file['name']).name
                fam_path = run_dir / Path(fam_file['name']).name
                shutil.copy(bed_file['datapath'], bed_path)
                shutil.copy(bim_file['datapath'], bim_path)
                shutil.copy(fam_file['datapath'], fam_path)

                # 以前缀（去掉 .bed 扩展）作为 --bfile 前缀
                prefix = Path(bed_file['name']).stem
                bfile_prefix = run_dir / prefix
                cmd += ["--bfile", str(bfile_prefix)]
            else:
                # TXT 特征：复制到专用目录，并用 --tfile 传绝对路径
                files = input.geno_file()
                if not files:
                    raise ValueError("Please upload a genotype TXT file / 请上传基因型TXT文件")
                geno_root = WORK_ROOT / "plink_uploads"
                geno_root.mkdir(parents=True, exist_ok=True)
                run_dir = geno_root / time.strftime("%Y%m%d-%H%M%S")
                run_dir.mkdir(exist_ok=True)
                geno_file = files[0]
                geno_path = run_dir / Path(geno_file["name"]).name
                shutil.copy(geno_file["datapath"], geno_path)
                cmd += ["--tfile", str(geno_path)]

            # DR 参数处理
            dr_methods = input.dr_methods()
            if dr_methods:
                # 防止同时选择 auto 和其他方法
                if "auto" in dr_methods and len(dr_methods) > 1:
                    raise ValueError("DR cannot select 'auto' and specific methods at the same time; keep only 'auto' or remove it / DR 参数不能同时选择 'auto' 和其他方法，请只保留 auto 或去掉 auto。")
                cmd += ["--DR", *dr_methods]

            # 验证集 Val 处理（与训练表型列号一致）
            if input.use_val():
                val_info = input.val_file()
                if not val_info:
                    raise ValueError("Please choose a validation phenotype file or uncheck the validation option / 请选择验证集表型文件或取消勾选验证集选项。")
                val_path = Path(val_info[0]['datapath'])
                cmd += ["--Val", str(val_path), "--Val-pos", pheno_pos]

            # 执行分析
            executor = AnalysisExecutor()
            result = await executor.execute(cmd=cmd)
            analysis_result.set(result)
            analysis_status.set("running" if result.get("success") else "failed")
            
        except Exception as e:
            # 添加错误提示
            ui.notification_show(f"Analysis failed / 分析失败: {str(e)}", duration=10, type="error")
            logging.exception("分析错误详情：")
            analysis_status.set("failed")

    # 确保结果变量已定义
    analysis_result = reactive.Value(None)
    
    # 新增下载处理器（同步版本）
    @output
    @render.download()
    def download_pred():
        """下载当前计算的预测结果"""
        if not analysis_result() or not analysis_result().get('success'):
            raise Exception("Please complete an analysis first / 请先完成计算")
        
        # 直接使用当前计算目录
        current_work_dir = Path(analysis_result().get('result_path'))
        pred_files = list(current_work_dir.glob("result_predict.txt"))
        if pred_files:
            return str(pred_files[0])
        raise Exception("Prediction file not found / 预测文件未找到")

    @output
    @render.download()
    def download_model():
        """下载当前计算的模型参数"""
        if not analysis_result() or not analysis_result().get('success'):
            raise Exception("Please complete an analysis first / 请先完成计算")
        
        current_work_dir = Path(analysis_result().get('result_path'))
        model_files = list(current_work_dir.glob("result_model.txt")) + list(current_work_dir.glob("result_model.keras"))
        if model_files:
            return str(model_files[0])
        raise Exception("Model file not found / 模型文件未找到")

    # 修正结果显示逻辑（删除重复定义）
    @output
    @render.ui
    @reactive.calc
    def result_log():
        status = analysis_status()  # 假设这是一个响应式变量
        
        # 如果还没有执行结果，优先返回友好提示，避免 NoneType 错误
        if not analysis_result() and status != "running":
            return ui.div("Waiting for analysis to start / 等待开始分析。", class_="text-muted")

        if status == "pending":
            return ui.div("Waiting for analysis to start... / 等待分析开始...", class_="text-muted")
        elif status == "running":
            try:
                result = analysis_result()
                if not result:
                    return ui.div("Starting analysis, please wait... / 正在启动分析，请稍候...", class_="text-info")
                log_path = Path(result.get('result_path')) / "IASML.log"
                if log_path.exists():
                    import subprocess
                    proc = subprocess.run(['tail', '-n', '100', str(log_path)], capture_output=True, text=True)
                    _log_content = proc.stdout
                    return ui.div(
                        ui.pre(
                            _log_content,
                            style="height: 400px; overflow-y: auto; background-color: #f8f9fa; padding: 10px;"
                        ),
                        class_="live-log"
                    )
                else:
                    return ui.div("Waiting for log file to be generated... / 正在等待日志文件生成...", class_="text-info")
            except Exception as e:
                return ui.div(f"Failed to read log / 日志读取失败: {str(e)}", class_="text-danger")
        elif status == "failed":
            return ui.div("The last analysis failed, please check inputs or try again / 最近一次分析失败，请检查输入或重试。", class_="text-danger")
        else:
            return ui.div("Unknown status / 未知状态！", class_="text-danger")

 # ========== 应用实例化 ==========
app = App(app_ui, server)

# ========== 启动配置 ========== 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=18001)
