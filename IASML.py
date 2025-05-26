import os
# 在导入任何库之前设置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽TensorFlow所有日志
os.environ['XGBOOST_SILENT'] = '1'  # 屏蔽XGBoost警告
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
# 过滤所有警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrame.applymap has been deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost", message=".*glibc.*")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="xgboost.core",  # 精确到子模块
    message=".*Your system has an old version of glibc.*manylinux2014.*"
)
# ... existing code ...
import random
import numpy as np
import pandas as pd
import sys
import time
import platform
import shutil
import argparse
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import gc
from pandas_plink import read_plink1_bin
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Lambda, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
import contextlib

# 设置日志
def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.FileHandler('IASML.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def process_chunk(chunk_data, chunk_snps, samples):
    """处理数据块的函数"""
    try:
        # 检查输入大小是否匹配
        if chunk_data.shape[0] != len(chunk_snps) or chunk_data.shape[1] != len(samples):
            raise ValueError(
                f"Data shape mismatch:chunk_data.shape={chunk_data.shape}, "
                f"chunk_snps length={len(chunk_snps)}, samples count={len(samples)}"
            )
        
        # 创建 DataFrame
        chunk_df = pd.DataFrame(chunk_data, index=chunk_snps, columns=samples)
        
        # 填充缺失值
        chunk_df = chunk_df.apply(lambda x: x.fillna(x.mean()), axis=0)
        
        # 交换 0 和 2 的位置
        chunk_df = chunk_df.applymap(lambda x: 2 if x == 0 else (0 if x == 2 else x))
        
        return chunk_df
    except Exception as e:
        logger.error(f"Error processing data chunk:{str(e)}")
        raise

def process_genotype_data(input_prefix, output_npy, chunk_size=10000, n_jobs=4):
    try:
        logger.info("Starting genotype format conversion...")

        # 读取PLINK文件
        G = read_plink1_bin(
            f'{input_prefix}.bed',
            f'{input_prefix}.bim',
            f'{input_prefix}.fam',
            verbose=False
        )

        # 获取样本ID和SNP信息
        samples = G.coords['sample'].values
        total_snps = len(G.coords['variant'])
        genotype_data = G.values

        # 读取bim文件获取SNP信息
        logger.info("Reading SNP information")
        bim = pd.read_csv(f'{input_prefix}.bim', delim_whitespace=True, header=None)

        # 生成SNP标识符 (chr:pos)
        snp_ids = [f'{row[0]}:{row[3]}' for row in bim.itertuples(index=False)]

        logger.info(f"Reading {len(samples)} samples with {total_snps} SNPs...")

        # 使用并行处理
        n_chunks = (total_snps + chunk_size - 1) // chunk_size

        def process_chunk_parallel(chunk_idx):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_snps)

            # 获取当前块的数据
            chunk_data = genotype_data[:, start_idx:end_idx]
            chunk_snps = snp_ids[start_idx:end_idx]

            # 创建DataFrame
            chunk_df = pd.DataFrame(
                chunk_data,
                index=samples,
                columns=chunk_snps
            )

            # 填充缺失值并转换编码
            chunk_df = chunk_df.apply(lambda x: x.fillna(x.mean()))
            chunk_df = chunk_df.applymap(lambda x: 2 if x == 0 else (0 if x == 2 else x))

            return chunk_df

        # 使用joblib进行并行处理
        logger.info(f"Processing data with {n_jobs} threads...")
        chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk_parallel)(i) 
            for i in tqdm(range(n_chunks), desc="Processing Genotype Data")
        )

        # 合并所有数据块
        logger.info("Merging data chunks...")
        final_df = pd.concat(chunks, axis=1)
        final_df.index.name = 'ID'

        # 保存结果为二进制文件
        logger.info("Merging the processed data...")
        np.save(output_npy, np.column_stack((samples, final_df.values)))
        logger.info(f"Data preparation completed. Results saved to {output_npy}")

    except Exception as e:
        logger.error(f"Error processing genotype data: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()

def process_phenotype_data_with_factors(phe_file, factor_columns, numeric_columns, phe_pos):
    logger.info("Processing phenotype data...")
    try:
        df = pd.read_csv(phe_file, sep='\t', header=0, index_col=0)
        logger.info(f"Phenotype file contains {df.shape[1]} traits")
        
        if factor_columns or numeric_columns:
            factor_columns = factor_columns or []
            numeric_columns = numeric_columns or []
            covariate_indices = factor_columns + numeric_columns
            logger.info(f"Processing covariate columns: {covariate_indices}")
            
            covariate_cols = [df.columns[i - 2] for i in covariate_indices if i - 2 < len(df.columns)]
            #logger.info(f"协变量列名: {covariate_cols}")
            
            incomplete_cases = df[df[covariate_cols].isna().any(axis=1) | (df[covariate_cols] == "").any(axis=1)].index
            complete_cases = df.index.difference(incomplete_cases)
            
            if len(incomplete_cases) > 0:
                logger.warning(f"The following individuals have missing covariates. {', '.join(incomplete_cases.astype(str))}")
            else:
                logger.info("No individuals with missing covariates found")
        else:
            complete_cases = df.index
            incomplete_cases = []

        processed_file = "processed_phenotype_data.txt"
        df.to_csv(processed_file, sep='\t', index=True)
        #logger.info(f"Adjusted phenotype data saved to: {processed_file}")
        
        return processed_file, incomplete_cases
    except Exception as e:
        logger.error(f"Error processing phenotype data: {str(e)}")
        raise

def preprocess_data(features_file, target_file, target_column, factor_columns=None, numeric_columns=None, is_txt=False, incomplete_cases=None):
    logger.info("Loading training data...")
    try:
        if is_txt:
            features_df = pd.read_csv(features_file, sep='\t', header=0, index_col=0)
            logger.info(f"Features loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features")
            feature_ids = features_df.index.astype(str)
            features_data = features_df.values
        else:
            features_data = np.load(features_file, allow_pickle=True)
            logger.info(f"Features loaded: {features_data.shape[0]} samples,  {features_data.shape[1] - 1} features")
            feature_ids = features_data[:, 0].astype(str)
            features_data = features_data[:, 1:]

        df = pd.read_csv(target_file, index_col=0, header=None, sep="\t")
        logger.info(f"Phenotype data loaded successfully")
        
        df.columns = [f"col_{i}" for i in range(1, df.shape[1] + 1)]
        target_col_name = df.columns[target_column - 2]
        
        df = df[df[target_col_name].notna() & (df[target_col_name] != "")]
        logger.info(f"Filter out individuals with missing phenotypes")
        
        if incomplete_cases is not None:
            incomplete_cases = np.array(list(incomplete_cases))  # 将 set 转换为 numpy 数组
            complete_cases = df.index.difference(incomplete_cases).astype(str)
            #logger.info(f"协变量缺失的个体数量: {len(incomplete_cases)}")
            if len(incomplete_cases) > 0:
                logger.warning(f"{len(incomplete_cases)} individuals have missing covariates")
        else:
            complete_cases = df.index.astype(str)
            incomplete_cases = np.array([])  # 确保是 numpy 数组

        index_r = np.intersect1d(feature_ids, complete_cases)
        index_p = np.setdiff1d(feature_ids, np.union1d(index_r, incomplete_cases))
        
        feature_r = features_data[np.isin(feature_ids, index_r)]
        feature_p = features_data[np.isin(feature_ids, index_p)]
        target_r = df.loc[index_r, target_col_name].values.astype(float)
        
        #logger.info(f"最终训练个体数: {len(index_r)}, 待预测个体数: {len(index_p)}")
        
        if feature_r.size == 0 or target_r.size == 0:
            logger.error(f"Size of the feature data: {feature_r.size}, Size of the target data: {target_r.size}")
            raise ValueError("Features or target data are empty. Please check the data loading and filtering steps.")
        
        X = feature_r.astype(float)
        y = target_r
        x = feature_p.astype(float)
        feature_names = df.columns[1:].tolist()
        
        logger.info(f"Data loaded: {len(index_r)} training samples, {len(index_p)} prediction samples with complete genotype and covariate information.")
        
        return X, y, feature_names, x, index_p, target_col_name
        
    except Exception as e:
        logger.error(f"Data preprocessing error: {str(e)}")
        raise

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Reshape((input_shape[0], 1)))  # 增加维度
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # 使用 Input 层指定输入形状
    model.add(Dense(96, activation='sigmoid'))  # 第一隐藏层
    model.add(Dense(64, activation='gelu'))     # 第二隐藏层
    model.add(Dense(1, activation='linear'))    # 输出层
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def get_model_and_params(model_type, input_shape, input_dim, ramsee, n_jobs):
    models = {
        'svm': (SVR(), {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }),
        'ridge': (Ridge(), {
            'alpha': np.logspace(-4, 0, 10),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }),
        'lasso': (Lasso(), {
            'alpha': np.logspace(-4, 0, 20)
        }),
        'elasticnet': (ElasticNet(), {
            'alpha': np.logspace(-4, 0, 10),
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }),
        'decision_tree': (DecisionTreeRegressor(random_state=ramsee), {
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }),
        'random_forest': (RandomForestRegressor(random_state=ramsee, n_jobs=n_jobs), {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }),
        'lightgbm': (LGBMRegressor(random_state=ramsee, n_jobs=n_jobs), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        'xgboost': (xgb.XGBRegressor(random_state=ramsee, n_jobs=n_jobs, enable_categorical=False,
            validate_parameters=True, # 添加显式继承声明
            **{'__sklearn_tags__': {'requires_y': True}}), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        'linear': (LinearRegression(n_jobs=n_jobs), {}),
        'pls': (PLSRegression(), {
            'n_components': np.arange(1, 100)
        }),
        'gbm': (GradientBoostingRegressor(random_state=ramsee), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        'cnn': (create_cnn_model(input_shape), {}),  # Keras 模型
        'mlp': (create_mlp_model(input_dim), {}),  # Keras 模型
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    model, params = models[model_type]
    
    # 确定模型是否为 scikit-learn 模型
    if model_type in ['cnn', 'mlp']:
        is_sklearn_model = False
    else:
        is_sklearn_model = True
    
    return model, params, is_sklearn_model

def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

class CustomEarlyStoppingAndCheckpoint(Callback):
    def __init__(self, patience=3, min_delta=0.1, model_path='best_model.keras'):
        super(CustomEarlyStoppingAndCheckpoint, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.Inf
        self.wait = 0
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is None:
            return

        print(f"\n  --  Epoch {epoch + 1}: current loss = {current_loss:.4f}, best loss = {self.best_loss:.4f}")

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            self.model.save(self.model_path)
            #print(f"Epoch {epoch + 1}: loss improved to {current_loss:.4f}, saving model to {self.model_path}")
        if  current_loss > self.best_loss:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEpoch {epoch + 1}: early stopping triggered")

def build_and_predict_sklearn(X, y, model, best_params, x, index_p, model_type, target_col_name):
    logger.info(f"Starting to train the {model_type} model...")
    try:
        # 设置模型参数
        model.set_params(**best_params)
        
        # 训练模型
        model.fit(X, y)
        
        # 进行预测
        Y_pred = model.predict(x)
        #save_predictions_to_file(Y_pred, target_col_name, f'{target_col_name}_{model_type}.txt', index_p)
        
        logger.info(f"{model_type} model training and prediction completed")
        return Y_pred
    except Exception as e:
        logger.error(f"Error training and predicting the {model_type} model: {str(e)}")
        raise

def build_and_predict_keras(X, y, x, index_p, model_type, target_col_name, out):
    logger.info(f"Starting to train the {model_type} model...")
    try:
        # 设置随机种子
        set_random_seed(42)
        
        # 创建模型
        if model_type == 'cnn':
            model = create_cnn_model((X.shape[1],))
        elif model_type == 'mlp':
            model = create_mlp_model(X.shape[1])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 早停机制
        early_stop = EarlyStopping(monitor='loss', patience=50)
        
        # 学习率递减
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        
        # 设置 batch_size 和 epochs
        batch_size = 32  # 您可以根据需要调整这个值
        epochs = 50  # 设置训练的epoch数
        
        # 训练模型
        model.fit(X, y, callbacks=[early_stop, reduce_lr, CustomEarlyStoppingAndCheckpoint()], batch_size=batch_size, epochs=epochs, verbose=1)
        best_model = load_model("best_model.keras", safe_mode=False)        
        Y_pred = best_model.predict(x)
        #save_predictions_to_file(Y_pred, target_col_name, f'{target_col_name}_{model_type}.txt', index_p)
        
        # 保存最终模型
        model_filename = f'{out}_final_model.keras'  # 使用h5格式
        best_model.save(model_filename)
        logger.info(f"The final model has been saved to {model_filename}")
        
        return Y_pred
    except Exception as e:
        logger.error(f"An error occurred during model training and prediction: {str(e)}")
        raise

def save_predictions_to_file(predictions, target_col_name, output_file, index_p):
    try:
        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=[target_col_name])
        predictions.index = index_p
        
        try:
            existing_data = pd.read_csv(output_file, sep="\t", index_col=0)
            combined_data = pd.concat([existing_data, predictions], axis=1)
        except FileNotFoundError:
            combined_data = predictions
            
        combined_data.to_csv(output_file, sep="\t")
        
        # 确保只记录一次日志
        #logger.info(f"预测结果已保存到 {output_file}")
    except Exception as e:
        logger.error(f"Error saving the prediction results: {str(e)}")
        raise

def split_phenotype_data(target_file, seed, n_splits=5):
    try:
        logger.info(f"Starting {n_splits}-fold cross-validation split...")
        
        # 读取表型数据
        df = pd.read_csv(target_file, sep='\t', header=0, index_col=0)
        logger.info(f"Phenotypic data loaded successfully, with shape: {df.shape}")
        
        # 初始化KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # 进行分割
        for fold, (train_index, val_index) in enumerate(kf.split(df), start=1):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]
            
            # 输出分割后的文件
            train_file = f"Ref{fold}.txt"
            val_file = f"Val{fold}.txt"
            train_df.to_csv(train_file, sep='\t')
            val_df.to_csv(val_file, sep='\t')
            
            logger.info(f"Fold {fold}: Training set saved to {train_file}, validation set saved to {val_file}")
        
    except Exception as e:
        logger.error(f"Error splitting the phenotypic data: {str(e)}")
        raise

def train_sklearn_model(X, y, model, best_params, model_type):
    logger.info(f"Starting to train the {model_type} model...")
    try:
        model.set_params(**best_params)
        model.fit(X, y)
        logger.info(f"{model_type} model training completed.")
        return model
    except Exception as e:
        logger.error(f"Error training the {model_type} model: {str(e)}")
        raise

def train_keras_model(X, y, x, index_p, model_type, target_col_name):
    logger.info(f"Starting to train the {model_type} model...")
    try:
        # 设置随机种子
        set_random_seed(42)
        
        # 创建模型
        input_dim = X.shape[1]
        if model_type == 'cnn':
            model = create_cnn_model((input_dim,))
        elif model_type == 'mlp':
            model = create_mlp_model(input_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}     ")
        
        # 模型检查点，保存最佳模型
        checkpoint = ModelCheckpoint(f"{target_col_name}_{model_type}_best_model.keras", save_best_only=True, monitor='loss', mode='min', verbose=1)
        
        # 早停机制
        early_stop = EarlyStopping(monitor='loss', patience=50)
        
        # 学习率递减
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        
        # 设置 batch_size 和 epochs
        batch_size = 32  
        epochs = 50  # 设置训练的epoch数
        
        model.fit(X, y, callbacks=[checkpoint, early_stop, reduce_lr, CustomEarlyStoppingAndCheckpoint()], batch_size=batch_size, epochs=epochs, verbose=1)
        Y_pred = model.predict(x)
        save_predictions_to_file(Y_pred, target_col_name, f'{target_col_name}_{model_type}.txt', index_p)
        
        # 保存最终模型
        model_filename = f'{target_col_name}_{model_type}_final_model.keras'
        model.save(model_filename)
        logger.info(f"The final model has been saved to {model_filename}")
        
        return Y_pred
    except Exception as e:
        logger.error(f"Error model training and prediction: {str(e)}")
        raise

def load_model_params_from_file(file_path):
    """从文件中加载模型参数"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            model_type = lines[0].strip()
            params = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    # 添加类型转换逻辑
                    try:
                        # 尝试转换为整数
                        params[key.strip()] = int(value.strip())
                    except ValueError:
                        try:
                            # 尝试转换为浮点数
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            # 保留字符串
                            params[key.strip()] = value.strip().strip("'\"")
                    # 处理布尔值
                    if params[key.strip()] == 'True':
                        params[key.strip()] = True
                    elif params[key.strip()] == 'False':
                        params[key.strip()] = False
                else:
                    logger.error(f"Incorrectly formatted line: {line.strip()}")
            return model_type, params
    except Exception as e:
        logger.error(f"Error loading the model parameter file: {str(e)}")
        raise

def save_model_params_to_file(model_type, params, file_path):
    """将模型参数保存到文件"""
    try:
        if params is None:
            logger.info(f"The {model_type} model has no parameters to save.")
            return

        with open(file_path, 'w') as f:
            f.write(f"{model_type}\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")  # 确保使用半角冒号
        logger.info(f"The model parameters have been saved to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the model parameter file: {str(e)}")
        raise
            
def main():
    # 获取平台信息
    platform_info = f"{platform.system()} {platform.machine()}"
    
    # 构建欢迎信息
    banner = f"""
#===========================================================================#
#                            Welcome to use IASML                           #
#                                                                           #
#  Version: v1.0.0 (2025-02-11)                                             #
#  Platform: {platform_info:12}              Developers: Linxi Zhu and Wentao Cai #
#  Homepage: iasbreeding.cn            Contact: ericisboy@163.com           #
#===========================================================================#
"""
    print(banner) 
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description=(
            "Machine Learning Package for Genomic Selection\n"
            "Core functionalities and required parameters:\n"
            "Genomic selection:--bfile, --phe, --phe-pos, --out, --model/--model-params\n"
            "Phenotype data splitting:--phe, --split-seed, --out\n"
            "Multi-omics prediction:--tfile, --phe, --phe-pos, --out, --model/--model-params\n"
            "Custom neural network:--bfile, --phe, --phe-pos, --out, --model-frame\n"
            "Other parameters are optional with default values."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        usage=argparse.SUPPRESS,  # 新增这行来隐藏usage信息
        add_help=False  # 禁用默认的-h帮助
    )
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='help')    
    parser.add_argument('--bfile', metavar='<fileprefix>', help="Prefix for genotype bed file (PLINK format)")
    parser.add_argument('--tfile', metavar='<filename>', help="Feature file name (txt format)")
    parser.add_argument('--phe', metavar='<filename>', required=True, help="Phenotype file name")
    parser.add_argument('--phe-pos', metavar='<col_number>', type=int, help="Column position of target phenotype")
    parser.add_argument('--f', metavar='<col_number>', default='', help="Factor covariates (comma-separated column numbers, default: none)")
    parser.add_argument('--n', metavar='<col_number>', default='', help="Numeric covariates (comma-separated column numbers, default: none)")
    parser.add_argument('--model', metavar='<model_type>', help=(
        "Model type, options:"
        "svm, ridge, lasso, elasticnet, decision_tree, random_forest, " 
        "lightgbm, xgboost, linear, pls, gbm, cnn, mlp"
    ))
    parser.add_argument('--model-params', metavar='<filename>', help="Parameter file for non-neural network models")
    parser.add_argument('--threads', metavar='<number>', type=int, default=8, help="Number of parallel threads (default: 8)")
    parser.add_argument('--n-iter', metavar='<number>', type=int, default=8, help="Number of parameter combinations for params random search (default: 8)")
    parser.add_argument('--cv-search', metavar='<number>', dest='cv_search', type=int, default=3, help="Cross-validation folds for params random search (default: 3)")
    parser.add_argument('--split-seed', metavar='<number>', type=int, help="Random seed for phenotype cross-validation splitting")
    parser.add_argument('--cv-split', dest='cv_split', metavar='<number>', type=int, default=5, choices=range(2, 11), help="Cross-validation folds for phenotype cross-validation splitting (2-10, default: %(default)d)")
    parser.add_argument('--model-frame', metavar='<filename>', help="Custom neural network model (keras format) for cnn/mlp")
    parser.add_argument('--out', required=True, metavar='<fileprefix>', help="Output file prefix")

    # 检查命令行参数数量
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # 如果提供了--split参数，则只执行数据分割
    if args.split_seed is not None:
        split_phenotype_data(
            target_file=args.phe,
            seed=args.split_seed,
             n_splits=args.cv_split  # 传递新的参数
        )
        logger.info("Phenotype splitting completed. Exiting program.")
        return

    # 检查其他必需参数
    if not args.bfile and not args.tfile:
        parser.error("--bfile or --tfile is required unless the --split-seed parameter is used.")
    if not (args.model_params or args.model or args.model_frame):
        parser.error("--model-params, --model or --model-frame are required unless the --split-seed parameter is used.")
    if args.phe_pos is None:
        parser.error("--phe-pos is required unless the --split-seed parameter is used.")

    n_jobs = args.threads
    ramsee = 42  # 定义随机种子

    try:
        start_time = time.time()
        
        # 确定特征文件
        if args.bfile:
            GENOTYPE = args.bfile
            output_npy = f"{GENOTYPE}.npy"
            process_genotype_data(GENOTYPE, output_npy, chunk_size=10000, n_jobs=n_jobs)
            features_file = output_npy
            is_txt = False
        else:
            features_file = args.tfile
            is_txt = True

        # 初始化协变量
        factor_columns = [int(x) for x in args.f.split(',') if x] if args.f else []
        numeric_columns = [int(x) for x in args.n.split(',') if x] if args.n else []

        # 校正协变量
        target_file, incomplete_cases = process_phenotype_data_with_factors(args.phe, factor_columns, numeric_columns, args.phe_pos)

        #logger.info(f"The phenotype file used: {target_file}")

        # 确保 target_file 是字符串
        if not isinstance(target_file, str):
            raise ValueError("target_file should be a string path.")

        # 数据预处理
        X, y, feature_names, x, index_p, target_col_name = preprocess_data(features_file, target_file, args.phe_pos, factor_columns, numeric_columns, is_txt, incomplete_cases)
        input_shape = (X.shape[1],)
        input_dim = X.shape[1]

        if args.model_frame:
            #if args.model not in ['cnn', 'mlp']:
                #parser.error("--model-frame 仅适用于 cnn 和 mlp 模型")
            logger.info(f"Load the pre-trained model: {args.model_frame}")
            model = load_model(args.model_frame)
            
            # 使用加载的模型进行预测
            Y_pred = model.predict(x)
            output_file = f'{args.out}_predict.txt'
            save_predictions_to_file(Y_pred, target_col_name, output_file, index_p)
            logger.info(f"Use the given model to make predictions and save to {output_file}.")
        else:
            if args.model_params and os.path.isfile(args.model_params):
                #if args.model in ['cnn', 'mlp']:
                    #parser.error("--model-params 不适用于 cnn 和 mlp 模型")
                model_type, best_params = load_model_params_from_file(args.model_params)
                logger.info(f"Model type loaded from file: {model_type}, parameters: {best_params}")
                model, _, is_sklearn_model = get_model_and_params(model_type, input_shape, input_dim, ramsee, n_jobs)
            else:
                model_type = args.model
                model, param_distributions, is_sklearn_model = get_model_and_params(model_type, input_shape, input_dim, ramsee, n_jobs)

                if is_sklearn_model:
                    # 对于 scikit-learn 模型，使用随机搜索进行参数调优
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_distributions,
                        n_iter=args.n_iter,
                        scoring='neg_mean_squared_error',
                        cv=args.cv_search,
                        n_jobs=n_jobs,
                        verbose=2,
                        random_state=ramsee
                    )
                    random_search.fit(X, y)
                    best_params = random_search.best_params_
                    logger.info(f"Best parameters: {best_params}")
                else:
                    # 对于 Keras 模型，不进行参数优化
                    best_params = None
                    logger.info(f"{model_type} does not require parameter optimization.")

            if is_sklearn_model:
                # 使用最佳参数进行训练和预测
                Y_pred = build_and_predict_sklearn(X, y, model, best_params, x, index_p, model_type, target_col_name)
            else:
                # 对于 Keras 模型，直接使用 Keras 的 API
                Y_pred = build_and_predict_keras(X, y, x, index_p, model_type, target_col_name, args.out)

            # 保存预测结果
            output_file = f'{args.out}_predict.txt'
            save_predictions_to_file(Y_pred, target_col_name, output_file, index_p)
            logger.info(f"The prediction results have been saved to {output_file}.")

            # 保存模型参数到文件
            params_file = f"{args.out}_model.txt"
            save_model_params_to_file(model_type, best_params, params_file)

        end_time = time.time()
        training_time = end_time - start_time
        # 清理所有可能产生的临时文件
        temp_files = [
            'processed_phenotype_data.txt',
            'best_model.keras',
            args.bfile + '.npy' if args.bfile else None  # 仅当使用--bfile时清理npy文件
        ]      
        for file in filter(None, temp_files):  # 过滤掉None值
            if os.path.exists(file):
                os.remove(file)
                #logger.info(f"Cleaned temporary file: {file}")
        logger.info(f"Total training time: {training_time:.4f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during the execution of the program: {str(e)}")
        # 清理所有可能产生的临时文件
        temp_files = [
            'processed_phenotype_data.txt',
            'best_model.keras',
            args.bfile + '.npy' if args.bfile else None  # 仅当使用--bfile时清理npy文件
        ]       
        for file in filter(None, temp_files):  # 过滤掉None值
            if os.path.exists(file):
                os.remove(file)
                #logger.info(f"Cleaned temporary file: {file}")
                
        sys.exit(1)

if __name__ == '__main__':
    main()
