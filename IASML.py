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
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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
        # 加载特征数据（携带index）
        if is_txt:
            features_df = pd.read_csv(features_file, sep='\t', header=0, index_col=0)
            logger.info(f"Features loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features")
            # 保持DataFrame格式，不转成values
            features_df_with_index = features_df
        else:
            features_data = np.load(features_file, allow_pickle=True)
            logger.info(f"Features loaded: {features_data.shape[0]} samples, {features_data.shape[1] - 1} features")
            # 创建带index的DataFrame
            feature_ids = features_data[:, 0].astype(str)
            features_data_values = features_data[:, 1:]
            features_df_with_index = pd.DataFrame(features_data_values, index=feature_ids)

        # 加载表型数据
        df = pd.read_csv(target_file, index_col=0, header=None, sep="\t")
        logger.info(f"Phenotype data loaded successfully")
        
        df.columns = [f"col_{i}" for i in range(1, df.shape[1] + 1)]
        target_col_name = df.columns[target_column - 2]
        
        # 过滤缺失表型
        df = df[df[target_col_name].notna() & (df[target_col_name] != "")]
        
        # 处理不完整案例
        if incomplete_cases is not None:
            incomplete_cases = set(incomplete_cases)
            complete_cases = df.index.difference(incomplete_cases).astype(str)
        else:
            complete_cases = df.index.astype(str)
            incomplete_cases = set()

        # 找出训练集和预测集的index（保持DataFrame索引操作）
        common_indices = features_df_with_index.index.intersection(complete_cases)
        index_r = common_indices  # 训练集index
        index_p = features_df_with_index.index.difference(index_r.union(incomplete_cases))  # 预测集index
        
        # 按index提取数据（确保顺序一致）
        feature_r_df = features_df_with_index.loc[index_r]  # 训练特征（带index）
        feature_p_df = features_df_with_index.loc[index_p]  # 预测特征（带index）
        target_r_series = df.loc[index_r, target_col_name]  # 训练表型（带index）
        
        # 验证顺序一致性
        print("验证顺序一致性:")
        print(f"特征index: {feature_r_df.index[:5].tolist()}")
        print(f"表型index: {target_r_series.index[:5].tolist()}")
        print(f"顺序一致: {feature_r_df.index.equals(target_r_series.index)}")
        
        # 只在最后一步提取values，确保转换为数值类型
        X = np.asarray(feature_r_df.values, dtype=np.float64)
        y = np.asarray(target_r_series.values, dtype=np.float64)
        x = np.asarray(feature_p_df.values, dtype=np.float64)
        
        feature_names = feature_r_df.columns.tolist() if hasattr(feature_r_df, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info(f"数据加载完成: {len(index_r)} 训练样本, {len(index_p)} 预测样本")
        logger.info(f"X形状: {X.shape}, y长度: {len(y)}")
        
        return X, y, feature_names, x, index_p, target_col_name
        
    except Exception as e:
        logger.error(f"数据预处理错误: {str(e)}")
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
            'alpha': np.logspace(-7, 0, 10),
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
            'n_estimators': [100, 200, 300, 400],               # 增加 n_estimators 的候选值
            'learning_rate': [0.01, 0.05, 0.1],                  # 尝试较小的学习率
            'max_depth': [3, 5, 7],                               # 树的深度
            'num_leaves': [31, 63, 127],                          # 增加叶子数的候选值
            'min_data_in_leaf': [20, 50, 100],                    # 最小叶子样本数
            'lambda_l1': [0.1, 1.0, 10],                          # L1 正则化
            'lambda_l2': [0.1, 1.0, 10],                          # L2 正则化
            'feature_fraction': [0.7, 0.8, 0.9, 1.0],            # 特征采样比例
            'bagging_fraction': [0.7, 0.8, 1.0],                  # 训练数据采样比例
            'bagging_freq': [1, 5, 10],                           # 数据采样频率
            'scale_pos_weight': [1, 2, 3]   
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

def _safe_standardize_rows(matrix):
    """按行进行零均值、单位方差标准化，避免零方差导致的除零问题。"""
    means = np.mean(matrix, axis=1, keepdims=True)
    stds = np.std(matrix, axis=1, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return (matrix - means) / stds

def van_raden_g_matrix(snp_matrix):
    """
    VanRaden (2008) 方法构建基因组关系矩阵 (G矩阵)
    公式: G = ZZ' / (2 * sum(p_i * (1-p_i)))
    其中 Z = M - P, M是基因型矩阵, P是等位基因频率矩阵
    """
    # 确保输入是numpy数组并转换为浮点型矩阵
    if not isinstance(snp_matrix, np.ndarray):
        snp_matrix = np.array(snp_matrix)
    M = np.asarray(snp_matrix, dtype=np.float64)
    
    # 计算每个SNP的等位基因频率 (p_i)
    p = np.nanmean(M, axis=0) / 2.0  # 假设是加性编码 (0,1,2)
    
    # 构建P矩阵 (每个元素是2*p_i)
    P = 2 * p
    
    # 计算Z矩阵: Z = M - P
    Z = M - P.reshape(1, -1)
    
    # 处理缺失值 (用0填充，即用期望值填充)
    Z = np.nan_to_num(Z, nan=0.0)
    
    # 计算分母: 2 * sum(p_i * (1-p_i))
    denominator = 2 * np.sum(p * (1 - p))
    
    # 避免除零错误
    if denominator < 1e-10:
        denominator = 1e-10
    
    # 计算G矩阵: G = ZZ' / denominator
    G_matrix = np.dot(Z, Z.T) / denominator
    
    return G_matrix

def yang_grm(snp_matrix):
    """
    Yang et al. 2010 方法构建基因组关系矩阵 (GRM)
    公式: GRM_ij = (1/M) * sum_{k=1}^M [(x_ik - 2p_k)(x_jk - 2p_k)] / [2p_k(1-p_k)]
    """
    # 确保输入是numpy数组并转换为浮点型矩阵
    if not isinstance(snp_matrix, np.ndarray):
        snp_matrix = np.array(snp_matrix)
    M = np.asarray(snp_matrix, dtype=np.float64)
    n_individuals, n_snps = M.shape
    
    # 计算每个SNP的等位基因频率
    p = np.nanmean(M, axis=0) / 2.0
    
    # 处理缺失值：用2p_k填充（期望值）
    missing_mask = np.isnan(M)
    if np.any(missing_mask):
        fill_matrix = 2 * p.reshape(1, -1)
        M = np.where(missing_mask, fill_matrix, M)
    
    # 中心化: x_ik - 2p_k
    Z = M - 2 * p.reshape(1, -1)
    
    # 计算权重: 1 / [2p_k(1-p_k)]
    weights = 1 / (2 * p * (1 - p))
    
    # 处理p_k=0或p_k=1的情况（避免除零）
    invalid_weights = np.isinf(weights) | np.isnan(weights)
    if np.any(invalid_weights):
        valid_mask = ~invalid_weights
        Z = Z[:, valid_mask]
        weights = weights[valid_mask]
        n_snps = int(np.sum(valid_mask))  # 确保是整数类型
    
    # 确保 n_snps 是有效的整数
    if n_snps <= 0:
        n_snps = 1
    
    # 应用权重
    Z_weighted = Z * np.sqrt(weights.reshape(1, -1))
    
    # 计算GRM: (1/M) * Z_weighted * Z_weighted^T
    GRM = np.dot(Z_weighted, Z_weighted.T) / float(n_snps)
    
    return GRM

def kl_divergence_similarity(X, Y=None):
    """
    基于KL散度计算个体相似度矩阵
    将每个个体的基因型转换为概率分布，然后计算KL散度
    KL散度越小，相似度越高，因此使用 exp(-KL) 作为相似度
    """
    if Y is None:
        Y = X
    
    # 确保输入是numpy数组并转换为浮点型
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    def rows_to_prob_dist(matrix):
        """将多行基因型转换为概率分布矩阵（向量化）"""
        # 处理缺失值
        matrix = np.nan_to_num(matrix, nan=0.0)
        n_rows, n_cols = matrix.shape
        
        # 计算每行的最小值和最大值
        row_mins = np.min(matrix, axis=1, keepdims=True)
        row_maxs = np.max(matrix, axis=1, keepdims=True)
        
        # 初始化概率矩阵
        prob_matrix = np.ones_like(matrix, dtype=float)
        
        # 处理每行：常数列使用均匀分布，非常数列使用softmax
        for i in range(n_rows):
            if row_maxs[i, 0] == row_mins[i, 0]:
                # 常数列使用均匀分布
                prob_matrix[i, :] = 1.0 / n_cols
            else:
                # 归一化
                normalized = (matrix[i, :] - row_mins[i, 0]) / (row_maxs[i, 0] - row_mins[i, 0] + 1e-10)
                # 使用softmax转换为概率分布
                exp_normalized = np.exp(normalized - np.max(normalized))
                prob_matrix[i, :] = exp_normalized / (np.sum(exp_normalized) + 1e-10)
        
        return prob_matrix
    
    # 转换为概率分布
    P_X = rows_to_prob_dist(X)
    P_Y = rows_to_prob_dist(Y)
    
    # 向量化计算KL散度矩阵
    # KL(P||Q) = sum(p * log(p/q))
    # 使用广播计算所有对
    log_ratio = np.log((P_X[:, np.newaxis, :] + 1e-10) / (P_Y[np.newaxis, :, :] + 1e-10))
    kl_matrix = np.sum(P_X[:, np.newaxis, :] * log_ratio, axis=2)
    
    # 转换为相似度: exp(-KL)
    similarity_matrix = np.exp(-kl_matrix)
    
    return similarity_matrix

def compute_similarity(X, Y=None, method='euclidean'):
    """根据给定方法计算个体相似度矩阵。
    支持: 'euclidean', 'cosine', 'hamming', 'manhattan', 'pearson', 'van_raden', 'yang_grm', 'kl_divergence'
    返回形状为 (n_samples_X, n_samples_Y or n_samples_X) 的相似度矩阵。
    """
    if Y is None:
        Y = X
    if method == 'euclidean':
        dist = pairwise_distances(X, Y, metric='euclidean')
        return 1.0 / (1.0 + dist)
    if method == 'manhattan':
        dist = pairwise_distances(X, Y, metric='manhattan')
        return 1.0 / (1.0 + dist)
    if method == 'hamming':
        dist = pairwise_distances(X, Y, metric='hamming')
        return 1.0 - dist
    if method == 'cosine':
        return cosine_similarity(X, Y)
    if method == 'pearson':
        # 行向量表示个体，按行标准化后计算相关系数
        Xz = _safe_standardize_rows(X)
        Yz = _safe_standardize_rows(Y)
        m = X.shape[1]
        if m <= 1:
            return np.zeros((X.shape[0], Y.shape[0]))
        return (Xz @ Yz.T) / (m - 1)
    if method == 'van_raden':
        # VanRaden G矩阵方法：需要先构建完整矩阵，然后提取需要的部分
        if Y is X or Y is None:
            return van_raden_g_matrix(X)
        else:
            # 对于X和Y不同的情况，需要合并后计算
            combined = np.vstack([X, Y])
            G_full = van_raden_g_matrix(combined)
            n_X = X.shape[0]
            return G_full[:n_X, n_X:]
    if method == 'yang_grm':
        # Yang GRM方法：需要先构建完整矩阵，然后提取需要的部分
        if Y is X or Y is None:
            return yang_grm(X)
        else:
            # 对于X和Y不同的情况，需要合并后计算
            combined = np.vstack([X, Y])
            GRM_full = yang_grm(combined)
            n_X = X.shape[0]
            return GRM_full[:n_X, n_X:]
    if method == 'kl_divergence':
        return kl_divergence_similarity(X, Y)
    raise ValueError(f"Unsupported DR method: {method}")

def compute_similarity_matrices(X_train, X_pred, method):
    """生成训练自相似矩阵和预测对训练的相似矩阵，基于(训练+预测)全集构建后再切分对齐。"""
    if X_pred is None or X_pred.size == 0:
        K_train = compute_similarity(X_train, None, method)
        return K_train, np.empty((0, K_train.shape[0]), dtype=K_train.dtype)
    combined = np.vstack([X_train, X_pred])
    K_full = compute_similarity(combined, combined, method)
    n_train = X_train.shape[0]
    # 训练-训练子块
    K_train = K_full[:n_train, :n_train]
    # 预测-训练子块（下半部左块）
    K_pred = K_full[n_train:, :n_train]
    return K_train, K_pred

def precompute_dr_kernels(X, x, methods, n_jobs=1):
    """并行预计算并缓存多种 DR 方法的核矩阵。返回 dict: method -> (K_train, K_pred)。"""
    methods = list(dict.fromkeys(methods))
    def _one(m):
        return m, compute_similarity_matrices(X, x, m)
    results = Parallel(n_jobs=n_jobs)(delayed(_one)(m) for m in methods)
    return {m: kp for m, kp in results}

def parse_dr_methods(dr_arg, dr_methods_all):
    """
    解析 --DR 参数，返回候选方法列表
    Args:
        dr_arg: args.DR 的值（可能是 None, 列表, 或单个字符串）
        dr_methods_all: 所有可用的 DR 方法列表
    Returns:
        candidate_methods: 候选方法列表，如果为 None 表示不使用 DR
    """
    if dr_arg is None:
        return None
    
    # 如果 dr_arg 是列表
    if isinstance(dr_arg, list):
        if len(dr_arg) == 0:
            return None
        # 检查是否包含 'auto'
        if 'auto' in dr_arg:
            if len(dr_arg) == 1:
                # 只有 'auto'，返回所有方法
                return dr_methods_all
            else:
                # 'auto' 和其他方法混合，视为错误
                raise ValueError("--DR 参数不能同时使用 'auto' 和其他方法。请使用 'auto' 或指定具体方法列表。")
        # 验证所有方法是否有效
        invalid_methods = [m for m in dr_arg if m not in dr_methods_all]
        if invalid_methods:
            raise ValueError(f"--DR 参数包含无效方法: {invalid_methods}。有效方法: {dr_methods_all}")
        return dr_arg
    
    # 如果 dr_arg 是单个字符串（向后兼容）
    if isinstance(dr_arg, str):
        if dr_arg == 'auto':
            return dr_methods_all
        elif dr_arg in dr_methods_all:
            return [dr_arg]
        else:
            raise ValueError(f"--DR 参数无效: {dr_arg}。有效方法: {dr_methods_all}")
    
    return None

def compute_parallel_configs(total_jobs, num_tasks):
    """根据总线程数与任务数，计算外层并行数与内层搜索线程数。"""
    if total_jobs is None or total_jobs <= 0:
        outer_jobs = num_tasks
        inner_jobs = 1
    else:
        outer_jobs = min(total_jobs, max(1, num_tasks))
        inner_jobs = max(1, total_jobs // outer_jobs)
    return outer_jobs, inner_jobs

def apply_pca_to_data(X_train, X_pred, variance_threshold=0.999, min_components=None):
    """
    对数据进行 PCA 降维（保留指定方差解释比例的主成分）。
    返回：X_train_transformed, X_pred_transformed, pca_model
    """
    n_samples = X_train.shape[0]
    # 限制主成分数不超过样本数
    if min_components is None:
        n_components_to_keep = n_samples
    else:
        n_components_to_keep = min(min_components, n_samples)
    
    pca = PCA(n_components=n_components_to_keep, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    # 计算需要保留的组件数以解释指定方差
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_needed = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    # 只保留所需的主成分
    X_train_final = X_train_pca[:, :n_components_needed]
    X_pred_final = pca.transform(X_pred)[:, :n_components_needed]
    
    logger.info(f"PCA 降维: 原始维度={X_train.shape[1]}, 保留主成分={n_components_needed}, 解释方差={cumsum_variance[n_components_needed-1]:.6f}")
    
    return X_train_final, X_pred_final, pca

def preprocess_for_model(X_train, X_pred, model_type):
    """根据模型类型对输入数据进行预处理（如 lightgbm 使用 PCA）"""
    if model_type == 'lightgbm':
        return apply_pca_to_data(X_train, X_pred, variance_threshold=0.999)
    return X_train, X_pred, None

def evaluate_model_cv(model, params, X, y, cv_folds):
    """在给定特征矩阵 X 上用固定参数进行 K 折交叉验证，返回平均负均方误差。"""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        try:
            mdl = model.__class__(**model.get_params())
            if params:
                valid_keys = set(mdl.get_params().keys())
                safe_params = {k: v for k, v in params.items() if k in valid_keys}
                mdl.set_params(**safe_params)
            mdl.fit(X_tr, y_tr)
            y_hat = mdl.predict(X_va)
            mse = mean_squared_error(y_va, y_hat)
            scores.append(-mse)
        except Exception:
            scores.append(-np.inf)
    return np.mean(scores) if len(scores) > 0 else -np.inf

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
        # 对 lightgbm 自动应用 PCA 降维（处理稀疏 SNP 特征）
        if model_type == 'lightgbm':
            X_use, x_use, pca_model = apply_pca_to_data(X, x, variance_threshold=0.999)
            logger.info(f"LightGBM PCA 降维: 特征从 {X.shape[1]} 降至 {X_use.shape[1]}")
        else:
            X_use, x_use = X, x
        
        # 设置模型参数（过滤掉模型不支持的键，如外部元信息 'DR' 等）
        if best_params is not None:
            valid_keys = set(model.get_params().keys())
            safe_params = {k: v for k, v in best_params.items() if k in valid_keys}
            model.set_params(**safe_params)
        
        # 训练模型
        model.fit(X_use, y)
        
        # 进行预测
        Y_pred = model.predict(x_use)
        #save_predictions_to_file(Y_pred, target_col_name, f'{target_col_name}_{model_type}.txt', index_p)
        pd.DataFrame(Y_pred).to_csv("pred.txt",index=True,header=None)
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
        # 对于 TensorFlow 2.10.x，自带的 tf.keras.load_model 不支持 safe_mode 参数
        best_model = load_model("best_model.keras")
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

def compute_and_log_validation_corr(val_file, index_p, target_col_name, Y_pred, out_prefix):
    try:
        df_val = pd.read_csv(val_file, index_col=0, header=None, sep='\t')
        df_val.columns = [f"col_{i}" for i in range(1, df_val.shape[1] + 1)]
        if target_col_name not in df_val.columns:
            # 与训练相同的列定位方式
            # 若 --phe-pos 为 k，则 target_col_name 为 col_{k-1}
            pass
        # 对齐到预测个体
        y_true = df_val.loc[df_val.index.intersection(index_p), target_col_name]
        y_true = y_true[y_true.notna() & (y_true != "")].astype(float)
        if y_true.empty:
            logger.warning("Validation set has no non-missing targets after alignment. Skip correlation.")
            return None
        y_pred_series = pd.Series(np.ravel(Y_pred), index=index_p)
        y_pred_aligned = y_pred_series.loc[y_true.index]
        if len(y_pred_aligned) != len(y_true):
            logger.warning("Aligned prediction and validation sizes differ; proceeding with intersection.")
        corr = float(np.corrcoef(y_true.values, y_pred_aligned.values)[0, 1])
        logger.info(f"Validation correlation (pred vs {target_col_name}): {corr:.6f}")
        try:
            with open(f"{out_prefix}_val_corr.txt", 'a', encoding='utf-8') as f:
                f.write(f"{target_col_name}\t{corr}\n")
        except Exception:
            pass
        return corr
    except Exception as e:
        logger.error(f"Error computing validation correlation: {str(e)}")
        return None

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
        if best_params is not None:
            valid_keys = set(model.get_params().keys())
            safe_params = {k: v for k, v in best_params.items() if k in valid_keys}
            model.set_params(**safe_params)
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
#  Version: v2.0.0 (2025-10-1)                                              #
#  Platform: {platform_info:12}        Developers: Linxi Zhu and Wentao Cai #
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
        "lightgbm, xgboost, linear, pls, gbm, cnn, mlp, gather"
    ))
    parser.add_argument(
        '--gather',
        nargs='*',
        metavar='<model_type>',
        help=(
            "Ensemble search over multiple base models. "
            "Provide one or more model names separated by space "
            "(e.g. --gather svm ridge random_forest), or 'all' to try all supported "
            "sklearn models. When --gather is used, it overrides --model and "
            "will automatically select the best-performing model (optionally combined "
            "with --DR) based on CV performance, then use that model for final training "
            "and prediction."
        )
    )
    parser.add_argument('--model-params', metavar='<filename>', help="Parameter file for non-neural network models")
    parser.add_argument('--threads', metavar='<number>', type=int, default=8, help="Number of parallel threads (default: 8)")
    parser.add_argument('--n-iter', metavar='<number>', type=int, default=32, help="Number of parameter combinations for params random search (default: 32)")
    parser.add_argument('--cv-search', metavar='<number>', dest='cv_search', type=int, default=3, help="Cross-validation folds for params random search (default: 3)")
    parser.add_argument('--split-seed', metavar='<number>', type=int, help="Random seed for phenotype cross-validation splitting")
    parser.add_argument('--cv-split', dest='cv_split', metavar='<number>', type=int, default=5, choices=range(2, 11), help="Cross-validation folds for phenotype cross-validation splitting (2-10, default: %(default)d)")
    parser.add_argument('--model-frame', metavar='<filename>', help="Custom neural network model (keras format) for cnn/mlp")
    parser.add_argument('--out', required=True, metavar='<fileprefix>', help="Output file prefix")
    parser.add_argument(
        '--DR',
        nargs='*',
        default=None,
        help=(
            "Dimension reduction by individual similarity (only supported for models: "
            "svm, ridge, lasso, elasticnet, random_forest, linear, pls, gbm). "
            "Use 'auto' to try all methods, or specify one or more methods: "
            "euclidean, cosine, hamming, manhattan, pearson, van_raden, yang_grm, kl_divergence"
        )
    )
    parser.add_argument('--val', metavar='<filename>', help="Optional validation phenotype file (same format as --phe) to compute prediction correlation")
    parser.add_argument('--Val', metavar='<filename>', help="External validation phenotype file (same format as --phe) for Pearson correlation")
    parser.add_argument('--Val-pos', metavar='<col_number>', type=int, help="Column position of validation phenotype in --Val (same rule as --phe-pos)")

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
    # 模型相关参数：允许使用 --model-params / --model / --model-frame / --gather 之一
    has_model_like = (args.model_params is not None) or (args.model is not None) or (args.model_frame is not None) or (getattr(args, "gather", None) is not None)
    if not has_model_like:
        parser.error("--model-params, --model, --model-frame or --gather is required unless the --split-seed parameter is used.")
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
        pd.DataFrame(X).to_csv('X_features.txt', index=False, header=False)
        pd.DataFrame(y).to_csv('y_target.txt', index=False, header=False)
        pd.DataFrame(x).to_csv('x_variable.txt', index=False, header=False)
        
        input_shape = (X.shape[1],)
        input_dim = X.shape[1]

        # DR: 个体相似矩阵替代原始特征
        chosen_dr_method = None
        dr_methods_all = ['euclidean', 'cosine', 'hamming', 'manhattan', 'pearson', 'van_raden', 'yang_grm', 'kl_divergence']
        candidate_dr_methods = parse_dr_methods(args.DR, dr_methods_all)
        if candidate_dr_methods:
            if len(candidate_dr_methods) == len(dr_methods_all) and (args.DR == ['auto'] or (isinstance(args.DR, list) and len(args.DR) == 1 and args.DR[0] == 'auto')):
                logger.info("--DR auto 模式：将在八种方法上分别进行评估/搜索并择优")
            elif len(candidate_dr_methods) == 1:
                logger.info(f"--DR 指定模式：使用 {candidate_dr_methods[0]}")
            else:
                logger.info(f"--DR 多方法模式：将在 {len(candidate_dr_methods)} 种方法上分别进行评估/搜索并择优: {candidate_dr_methods}")

        if args.model_params and args.model:
            logger.warning("--model-params 已提供，将优先使用参数文件，忽略 --model 选项。")

        if args.model_frame:
            #if args.model not in ['cnn', 'mlp']:
                #parser.error("--model-frame 仅适用于 cnn 和 mlp 模型")
            logger.info(f"Load the pre-trained model: {args.model_frame}")
            model = load_model(args.model_frame)
            
            if candidate_dr_methods:
                logger.warning("--DR 参数对自定义 Keras 模型 (cnn/mlp) 不生效，已忽略。")

            # 使用加载的模型进行预测
            Y_pred = model.predict(x)
            output_file = f'{args.out}_predict.txt'
            save_predictions_to_file(Y_pred, target_col_name, output_file, index_p)
            logger.info(f"Use the given model to make predictions and save to {output_file}.")
        else:
            if args.model_params and os.path.isfile(args.model_params):
                # 读取第一行判断模型/策略
                with open(args.model_params, 'r') as f:
                    lines_all = [ln.strip() for ln in f]
                header = lines_all[0].lower() if lines_all else ''
                if header == 'gather':
                    # 构造候选模型列表
                    rest = [ln for ln in lines_all[1:] if ln]
                    candidate_model_types = rest if rest else ['svm', 'ridge', 'lasso', 'elasticnet', 'random_forest', 'linear', 'pls']
                    logger.info(f"Gather候选模型(来自文件): {candidate_model_types}")

                    # 使用统一的 DR 方法解析
                    gather_candidate_dr_methods = parse_dr_methods(args.DR, dr_methods_all)
                    if gather_candidate_dr_methods is None:
                        gather_candidate_dr_methods = [None]

                    best_global = {'score': -np.inf, 'model_type': None, 'params': None, 'DR': None, 'X': None, 'x': None}
                    for mtype in candidate_model_types:
                        mdl, param_distributions, is_sklearn_model = get_model_and_params(mtype, input_shape, input_dim, ramsee, n_jobs)
                        if not is_sklearn_model:
                            logger.warning(f"模型 {mtype} 为 Keras/不支持 gather，跳过。")
                            continue
                        for drm in gather_candidate_dr_methods:
                            if drm:
                                K_train, K_pred = compute_similarity_matrices(X, x, drm)
                                X_try, x_try = K_train, K_pred
                            else:
                                X_try, x_try = X, x
                            rs = RandomizedSearchCV(
                                estimator=mdl,
                                param_distributions=param_distributions,
                                n_iter=args.n_iter,
                                scoring='neg_mean_squared_error',
                                cv=args.cv_search,
                                n_jobs=n_jobs,
                                verbose=0,
                                random_state=ramsee
                            )
                            rs.fit(X_try, y)
                            score = rs.best_score_
                            logger.info(f"gather候选: model={mtype}, DR={drm or 'none'}, best_score={score:.6f}, best_params={rs.best_params_}")
                            if score > best_global['score']:
                                best_global = {'score': score, 'model_type': mtype, 'params': rs.best_params_, 'DR': drm, 'X': X_try, 'x': x_try}
                    if best_global['model_type'] is None:
                        raise RuntimeError("gather 未找到有效候选模型。")
                    chosen_dr_method = best_global['DR']
                    X, x = best_global['X'], best_global['x']
                    model_type = best_global['model_type']
                    model, _, is_sklearn_model = get_model_and_params(model_type, (X.shape[1],), X.shape[1], ramsee, n_jobs)
                    best_params = best_global['params']
                    if chosen_dr_method:
                        best_params = {**best_params, 'DR': chosen_dr_method}
                    input_shape = (X.shape[1],)
                    input_dim = X.shape[1]
                    logger.info(f"gather选择: model={model_type}, DR={chosen_dr_method or 'none'}, score={best_global['score']:.6f}")
                else:
                    # 普通参数文件：第一行是模型名，后续键值参数
                    model_type, best_params = load_model_params_from_file(args.model_params)
                    logger.info(f"Model type loaded from file: {model_type}, parameters: {best_params}")
                    model, _, is_sklearn_model = get_model_and_params(model_type, input_shape, input_dim, ramsee, n_jobs)
                    # 若提供固定参数，也支持 DR 作为外部选择
                    if candidate_dr_methods:
                        if is_sklearn_model:
                            candidate_methods = candidate_dr_methods
                            best_score = -np.inf
                            best_pair = None
                            best_X = None
                            best_xpred = None
                            for mth in candidate_methods:
                                K_train, K_pred = compute_similarity_matrices(X, x, mth)
                                score = evaluate_model_cv(model, best_params, K_train, y, args.cv_search)
                                logger.info(f"DR={mth}，固定参数的CV得分(neg MSE)={score:.6f}")
                                if score > best_score:
                                    best_score = score
                                    best_pair = mth
                                    best_X = K_train
                                    best_xpred = K_pred
                            if best_pair is None:
                                raise RuntimeError("DR auto 评估失败，未得到有效得分")
                            chosen_dr_method = best_pair
                            X, x = best_X, best_xpred
                            input_shape = (X.shape[1],)
                            input_dim = X.shape[1]
                            # 记录所选 DR
                            if best_params is None:
                                best_params = {'DR': chosen_dr_method}
                            else:
                                best_params = {**best_params, 'DR': chosen_dr_method}
                            logger.info(f"选择的DR方法: {chosen_dr_method}")
                        else:
                            # Keras 模型不支持 DR，忽略
                            logger.warning("--DR 参数对 Keras 模型 (cnn/mlp) 不生效，已忽略。")
            else:
                # =========================
                # 普通模型 / gather 集成策略
                # =========================
                # 若提供了 --gather，则优先走 gather 流程；否则使用 --model
                gather_cli = args.gather if hasattr(args, 'gather') else None
                model_type = args.model

                # 判断是否启用 gather：
                # 1) 显式指定 --model gather
                # 2) 或者提供了 --gather 模型列表
                use_gather = (model_type == 'gather') or (gather_cli is not None and len(gather_cli) > 0)

                if use_gather:
                    # 所有支持 gather（基于 sklearn）的模型列表
                    all_gather_models = [
                        'svm', 'ridge', 'lasso', 'elasticnet',
                        'decision_tree', 'random_forest',
                        'lightgbm', 'xgboost',
                        'linear', 'pls', 'gbm'
                    ]

                    if gather_cli is not None and len(gather_cli) > 0:
                        if 'all' in gather_cli:
                            candidate_model_types = all_gather_models
                        else:
                            invalid = [m for m in gather_cli if m not in all_gather_models]
                            if invalid:
                                raise ValueError(
                                    f"--gather 中包含不支持的模型: {invalid}。"
                                    f"可选模型: {all_gather_models}。"
                                )
                            candidate_model_types = gather_cli
                        logger.info(f"Gather候选模型(来自 --gather): {candidate_model_types}")
                    else:
                        # 退回到旧行为：--model gather 但未提供 --gather 时的默认模型集合
                        candidate_model_types = all_gather_models
                        logger.info(f"Gather候选模型(来自 --model gather 默认): {candidate_model_types}")

                    # 使用统一的 DR 方法解析（gather 可以与 --DR 联用）
                    gather_candidate_dr_methods = parse_dr_methods(args.DR, dr_methods_all)
                    if gather_candidate_dr_methods is None:
                        gather_candidate_dr_methods = [None]

                    best_global = {
                        'score': -np.inf,
                        'model_type': None,
                        'params': None,
                        'DR': None,
                        'X': None,
                        'x': None
                    }

                    # 预计算核缓存（仅当需要）
                    kernels_needed = [m for m in gather_candidate_dr_methods if m]
                    outer_jobs_kernels, _ = compute_parallel_configs(n_jobs, len(kernels_needed) if kernels_needed else 1)
                    kernel_cache = precompute_dr_kernels(X, x, kernels_needed, n_jobs=outer_jobs_kernels) if kernels_needed else {}

                    # 并行在 (model, DR) 组合上搜索（动态分配并行）
                    combos = [(mtype, drm) for mtype in candidate_model_types for drm in gather_candidate_dr_methods]
                    outer_jobs, inner_jobs = compute_parallel_configs(n_jobs, len(combos) if len(combos) > 0 else 1)
                    logger.info(f"gather并行配置: 外层任务={outer_jobs}, 内层每任务线程={inner_jobs}")

                    def _search_combo(mtype, drm):
                        mdl, param_distributions, is_sklearn_model = get_model_and_params(mtype, input_shape, input_dim, ramsee, n_jobs)
                        if not is_sklearn_model:
                            return mtype, drm, -np.inf, None, None, None
                        if drm:
                            K_train, K_pred = kernel_cache[drm]
                            X_try, x_try = K_train, K_pred
                        else:
                            X_try, x_try = X, x
                        # 对 lightgbm 应用 PCA 预处理
                        X_fit, x_pred, _ = preprocess_for_model(X_try, x_try, mtype)
                        rs = RandomizedSearchCV(
                            estimator=mdl,
                            param_distributions=param_distributions,
                            n_iter=args.n_iter,
                            scoring='neg_mean_squared_error',
                            cv=args.cv_search,
                            n_jobs=inner_jobs,
                            verbose=0,
                            random_state=ramsee
                        )
                        rs.fit(X_fit, y)
                        return mtype, drm, rs.best_score_, rs.best_params_, X_try, x_try

                    results = Parallel(n_jobs=outer_jobs)(delayed(_search_combo)(mtype, drm) for mtype, drm in combos)
                    for mtype, drm, score, params_found, X_try, x_try in results:
                        if params_found is None:
                            continue
                        logger.info(f"gather候选: model={mtype}, DR={drm or 'none'}, best_score={score:.6f}, best_params={params_found}")
                        if score > best_global['score']:
                            best_global = {'score': score, 'model_type': mtype, 'params': params_found, 'DR': drm, 'X': X_try, 'x': x_try}

                    if best_global['model_type'] is None:
                        raise RuntimeError("gather 未找到有效候选模型。")

                    # 采用最佳配置
                    chosen_dr_method = best_global['DR']
                    X, x = best_global['X'], best_global['x']
                    model_type = best_global['model_type']
                    model, _, is_sklearn_model = get_model_and_params(model_type, (X.shape[1],), X.shape[1], ramsee, n_jobs)
                    best_params = best_global['params']
                    if chosen_dr_method:
                        best_params = {**best_params, 'DR': chosen_dr_method}
                    input_shape = (X.shape[1],)
                    input_dim = X.shape[1]
                    logger.info(f"gather选择: model={model_type}, DR={chosen_dr_method or 'none'}, score={best_global['score']:.6f}")
                    # gather 流程已完成参数搜索，跳过后续参数搜索逻辑
                    param_distributions = None
                else:
                    model, param_distributions, is_sklearn_model = get_model_and_params(model_type, input_shape, input_dim, ramsee, n_jobs)

                if is_sklearn_model and param_distributions is not None:
                    # 对于 scikit-learn 模型，使用随机搜索进行参数调优
                    if candidate_dr_methods:
                        candidate_methods = candidate_dr_methods
                        # 并行预计算核
                        outer_jobs_kernels, _ = compute_parallel_configs(n_jobs, len(candidate_methods))
                        kernel_cache = precompute_dr_kernels(X, x, candidate_methods, n_jobs=outer_jobs_kernels)
                        best_global_score = -np.inf
                        best_global_params = None
                        best_global_method = None
                        best_global_X = None
                        best_global_xpred = None
                        # 并行搜索不同方法，动态分配外层/内层并行
                        outer_jobs, inner_jobs = compute_parallel_configs(n_jobs, len(candidate_methods))
                        logger.info(f"DR搜索并行配置: 外层任务={outer_jobs}, 内层每任务线程={inner_jobs}")
                        def _search_one(mth):
                            K_train, K_pred = kernel_cache[mth]
                            rs = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=param_distributions,
                                n_iter=args.n_iter,
                                scoring='neg_mean_squared_error',
                                cv=args.cv_search,
                                n_jobs=inner_jobs,
                                verbose=0,
                                random_state=ramsee
                            )
                            rs.fit(K_train, y)
                            return mth, rs.best_score_, rs.best_params_, K_train, K_pred
                        results = Parallel(n_jobs=outer_jobs)(delayed(_search_one)(m) for m in candidate_methods)
                        for mth, score, params_found, K_train, K_pred in results:
                            logger.info(f"DR={mth} 的最佳CV得分(neg MSE)={score:.6f}，最佳参数={params_found}")
                            if score > best_global_score:
                                best_global_score = score
                                best_global_params = params_found
                                best_global_method = mth
                                best_global_X = K_train
                                best_global_xpred = K_pred
                        if best_global_method is None:
                            raise RuntimeError("DR auto 搜索失败，未得到有效结果")
                        chosen_dr_method = best_global_method
                        X, x = best_global_X, best_global_xpred
                        best_params = {**best_global_params, 'DR': chosen_dr_method}
                        input_shape = (X.shape[1],)
                        input_dim = X.shape[1]
                        logger.info(f"最终选择 DR 方法: {chosen_dr_method}，并附加到最佳参数中")
                    else:
                        # 指定DR或未启用DR：先行转换（若有），再进行搜索
                        if candidate_dr_methods and len(candidate_dr_methods) == 1:
                            # 单个方法，直接使用
                            chosen_dr_method = candidate_dr_methods[0]
                            K_train, K_pred = compute_similarity_matrices(X, x, chosen_dr_method)
                            X, x = K_train, K_pred
                            input_shape = (X.shape[1],)
                            input_dim = X.shape[1]
                            logger.info(f"使用指定DR方法: {chosen_dr_method}")
                        elif candidate_dr_methods and len(candidate_dr_methods) > 1:
                            # 多个方法，需要评估并选择最佳的
                            outer_jobs_kernels, _ = compute_parallel_configs(n_jobs, len(candidate_dr_methods))
                            kernel_cache = precompute_dr_kernels(X, x, candidate_dr_methods, n_jobs=outer_jobs_kernels)
                            best_score = -np.inf
                            best_method = None
                            best_X = None
                            best_xpred = None
                            for mth in candidate_dr_methods:
                                K_train, K_pred = kernel_cache[mth]
                                score = evaluate_model_cv(model, None, K_train, y, args.cv_search)
                                logger.info(f"DR={mth}，CV得分(neg MSE)={score:.6f}")
                                if score > best_score:
                                    best_score = score
                                    best_method = mth
                                    best_X = K_train
                                    best_xpred = K_pred
                            if best_method is None:
                                raise RuntimeError("DR 多方法评估失败，未得到有效结果")
                            chosen_dr_method = best_method
                            X, x = best_X, best_xpred
                            input_shape = (X.shape[1],)
                            input_dim = X.shape[1]
                            logger.info(f"从 {len(candidate_dr_methods)} 种方法中选择最佳: {chosen_dr_method}")
                        random_search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_distributions,
                            n_iter=args.n_iter,
                            scoring='neg_mean_squared_error',
                            cv=args.cv_search,
                            n_jobs=max(1, n_jobs // 2),
                            verbose=2,
                            random_state=ramsee
                        )
                        random_search.fit(X, y)
                        best_params = random_search.best_params_
                        # 同步更新模型为最佳估计器，确保后续训练与搜索一致
                        try:
                            model = random_search.best_estimator_
                        except Exception:
                            pass
                        if chosen_dr_method:
                            best_params = {**best_params, 'DR': chosen_dr_method}
                        logger.info(f"Best parameters: {best_params}")
                else:
                    # 对于 Keras 模型，不进行参数优化
                    best_params = None
                    logger.info(f"{model_type} does not require parameter optimization.")
                    # Keras：若启用DR，忽略
                    if candidate_dr_methods:
                        logger.warning("--DR 参数对 Keras 模型 (cnn/mlp) 不生效，已忽略。")

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

            # 如果提供验证集，计算相关系数
            if args.val:
                compute_and_log_validation_corr(args.val, index_p, target_col_name, Y_pred, args.out)

            # 保存模型参数到文件
            params_file = f"{args.out}_model.txt"
            save_model_params_to_file(model_type, best_params, params_file)

        end_time = time.time()
        training_time = end_time - start_time

        # 如果提供了新的外部验证集参数 (--Val 和 --Val-pos)，计算 Pearson 相关并输出包含运行时间的结果文件
        if args.Val is not None and args.Val_pos is not None:
            try:
                df_val_ext = pd.read_csv(args.Val, index_col=0, header=None, sep='\t')
                df_val_ext.columns = [f"col_{i}" for i in range(1, df_val_ext.shape[1] + 1)]
                # 与训练数据相同的列定位规则：实际表型列索引为 (Val-pos - 2)
                val_col_name = df_val_ext.columns[args.Val_pos - 2]
                # 对齐到预测个体
                y_true_ext = df_val_ext.loc[df_val_ext.index.intersection(index_p), val_col_name]
                y_true_ext = y_true_ext[y_true_ext.notna() & (y_true_ext != "")].astype(float)
                if not y_true_ext.empty:
                    y_pred_series_ext = pd.Series(np.ravel(Y_pred), index=index_p)
                    y_pred_aligned_ext = y_pred_series_ext.loc[y_true_ext.index]
                    pearson_r = float(np.corrcoef(y_true_ext.values, y_pred_aligned_ext.values)[0, 1])
                    out_file = f"{args.out}_Val_pearson.txt"
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write("trait\tpearson_r\truntime_seconds\n")
                        f.write(f"{val_col_name}\t{pearson_r}\t{training_time:.6f}\n")
                    logger.info(f"External validation Pearson correlation saved to {out_file}: r = {pearson_r:.6f}, runtime = {training_time:.4f} seconds")
                else:
                    logger.warning("External validation set has no non-missing targets after alignment. Skip Pearson correlation output.")
            except Exception as e:
                logger.error(f"Error computing external validation Pearson correlation: {str(e)}")

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
