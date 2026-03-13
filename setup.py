from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    # 允许以 "#" 开头的注释行，但会在 install_requires 中自动过滤
    raw_reqs = [line.strip() for line in f if line.strip()]
    requirements = [r for r in raw_reqs if not r.startswith("#")]

setup(
    name="iasml",
    version="2.0.0",
    author="Linxi Zhu and Wentao Cai",
    author_email="ericisboy@163.com",
    description="Integrated Analysis System for Machine Learning in Genomic Selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://iasbreeding.cn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "iasml=IASML:main",  # 假设你的主函数在 IASML.py 中
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
