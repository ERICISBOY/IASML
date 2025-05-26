from setuptools import setup

setup(
    name="IASML",
    version="1.0",
    py_modules=["IASML"],  # 直接使用单个文件
    install_requires=[  # 依赖的包
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn',
        'tensorflow',
        'xgboost',
        'lightgbm',
        'joblib',
        'tqdm',
        'statsmodels',
        'pandas-plink',
        'scikeras',
        'keras-tuner',
        'h5py',
        'pyarrow',
        'dask',
        'scipy',
        'keras'
    ],
    entry_points={
        'console_scripts': [
            ' IASML=IASML:main',  # 假设主函数是main()
        ],
    },
)