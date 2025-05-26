let apiBase = process.env.NODE_ENV === 'development' 
    ? 'http://localhost:5000' 
    : '/api';

// 文件上传处理
async function handleFileUpload(fileType) {
    const formData = new FormData();
    const files = {
        bed: document.getElementById('bedFile').files[0],
        bim: document.getElementById('bimFile').files[0],
        fam: document.getElementById('famFile').files[0],
        txt: document.getElementById('txtFile').files[0]
    };

    // 根据选择的文件类型上传
    if (fileType === 'txt') {
        if (files.txt) formData.append('txt', files.txt);
    } else {
        ['bed', 'bim', 'fam'].forEach(prefix => {
            if (files[prefix]) formData.append(prefix, files[prefix]);
        });
    }

    try {
        const response = await fetch(`${apiBase}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            // 添加文件类型到分析请求
            result.file_type = fileType; 
            return result;
        }
        throw new Error(result.message);
    } catch (error) {
        showError(`文件上传错误: ${error.message}`);
        throw error;
    }
}

// 执行分析
async function runAnalysis() {
    const modelType = document.getElementById('modelType').value;
    const fileType = document.querySelector('input[name="fileType"]:checked').value;
    
    try {
        const uploadResult = await handleFileUpload(fileType);
        
        const analysisResponse = await fetch(`${apiBase}/analyze`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: uploadResult.session_id,
                modelType: modelType,
                file_type: uploadResult.file_type  // 传递文件类型
            })
        });
        
        const result = await analysisResponse.json();
        if (result.status === 'success') {
            displayResults(result);
            enableDownload(result.output_dir); // 新增下载功能
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        showError(`分析错误: ${error.message}`);
    }
}

// 显示结果
function displayResults(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <h3>分析结果</h3>
        <div class="result-card">
            <p>准确率: ${data.accuracy}%</p>
            <pre>${JSON.stringify(data.metrics, null, 2)}</pre>
        </div>
    `;
}

// 修改文件读取添加校验
async function readFile(elementId) {
    const fileInput = document.getElementById(elementId);
    if (!fileInput.files || fileInput.files.length === 0) {
        throw new Error('请先选择所有必需文件');
    }
    const file = fileInput.files[0];
    // 添加文件类型校验
    const allowedTypes = {
        'bedFile': ['bed'],
        'bimFile': ['bim'],
        'famFile': ['fam']
    };
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowedTypes[elementId].includes(ext)) {
        throw new Error(`无效的${elementId}文件类型`);
    }
    // 增强文件头校验
    const header = new Uint8Array(await file.slice(0, 2).arrayBuffer());
    if (elementId === 'bedFile' && !(header[0] === 0x6C && header[1] === 0x1B)) {
        throw new Error('无效的BED文件格式');
    }
    // 补充文件扩展名校验
    const validExtensions = {
        'bedFile': '.bed',
        'bimFile': '.bim', 
        'famFile': '.fam'
    };
    if (!file.name.endsWith(validExtensions[elementId])) {
        throw new Error(`文件扩展名必须为${validExtensions[elementId]}`);
    }
    return new Uint8Array(await file.arrayBuffer());
}

// 结果下载
function downloadResult() {
    try {
        const content = pyodide.FS.readFile('result_predict.txt');
        const blob = new Blob([content], {type: 'text/plain;charset=utf-8'});
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `IASML_Result_${new Date().toISOString().slice(0,10)}.txt`;
        link.click();
        document.getElementById('downloadBtn').style.display = 'none';
    } catch (error) {
        alert('结果文件生成失败，请重新运行分析');
    }
}

// 添加全局状态追踪
let docInitialized = false;

let docLoaded = false;

// 修改switchTab函数（删除文档加载相关逻辑）
function switchTab(tabId) {
    // 隐藏所有tab内容
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none'; 
    });

    // 显示目标tab
    const targetTab = document.getElementById(tabId);
    if(targetTab) {
        targetTab.style.display = 'block';
    }

    // 更新按钮状态
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.currentTarget.classList.add('active');
}

// 添加卡片悬停效果
document.querySelectorAll('.download-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.boxShadow = '0 12px 30px rgba(0,0,0,0.12)';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.boxShadow = '0 6px 20px rgba(0,0,0,0.08)';
    });
});

// 保留增强后的下载函数
function downloadPackage(type) {
    // 修正后的下载路径
    const packages = {
        core: {
            url: 'IASML-1.0.0.tar.gz',
            filename: 'IASML-1.0.0.tar.gz'
        },
        sample: {
            url: 'IASML-Windows-1.0.0.rar',
            filename: 'IASML-Windows-1.0.0.rar'
        }
    };

    // 创建隐藏的下载链接
    const link = document.createElement('a');
    link.href = packages[type].url;
    link.download = packages[type].filename;
    link.style.display = 'none';
    
    // 添加到DOM并触发点击
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // 添加下载动画
    event.target.classList.add('downloading');
    setTimeout(() => {
        event.target.classList.remove('downloading');
    }, 1000);
}

// 添加滚动高亮
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.doc-card, .download-header');
    const navLinks = document.querySelectorAll('.floating-nav a');
    
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });
});

// 修正后的锚点跳转
document.querySelectorAll('.model-tag').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const target = document.querySelector(targetId);
        
        if (target) {
            // 确保目标可见
            target.style.opacity = '1';
            target.style.height = 'auto';
            
            // 滚动到目标位置
            window.scrollTo({
                top: target.offsetTop - 100,
                behavior: 'smooth'
            });
        }
    });
});

// 自动渲染数学公式
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
};

// URL锚点自动高亮
window.addEventListener('hashchange', () => {
    const target = document.querySelector(location.hash);
    if(target) {
        target.style.transition = 'none';
        target.style.boxShadow = '0 0 15px rgba(109,139,157,0.2)';
        setTimeout(() => {
            target.style.transition = 'all 0.3s';
            target.style.boxShadow = '';
        }, 2000);
    }
});

// 新增下载功能
function enableDownload(outputDir) {
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'block';
    downloadBtn.onclick = () => {
        window.open(`${apiBase}/download?path=${encodeURIComponent(outputDir)}`);
    };
}