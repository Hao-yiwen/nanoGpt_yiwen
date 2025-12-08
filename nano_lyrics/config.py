# -*- coding: utf-8 -*-
"""
NanoGPT 配置文件

所有超参数集中管理
"""

import torch

# ============================================================================
# 数据配置
# ============================================================================

LYRICS_DIR = '../ChineseLyrics'           # 歌词数据目录（相对于 final_train/）
BPE_VOCAB_SIZE = 10000                     # BPE 词汇表大小
BPE_MODEL_PATH = 'chinese_lyrics_bpe.json' # BPE 模型文件路径
MAX_SONGS = 102197                         # 加载的歌曲数量
RANDOM_SEED = 1337

# ChatML 特殊 token 配置
SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "eos_token": "<|endoftext|>",
    "im_start_token": "<|im_start|>",
    "im_end_token": "<|im_end|>",
}

# ============================================================================
# 模型架构 (GPT-2 Medium 规格)
# ============================================================================

N_EMBED = 1024           # 嵌入维度
N_HEADS = 16             # 注意力头数 (1024 / 16 = 64, 偶数满足 RoPE)
N_KV_HEADS = 4           # GQA 的 KV 头数（必须能整除 N_HEADS，可选: 1, 2, 4, 8, 16）
N_LAYERS = 24            # Transformer Block 层数
DROP_OUT = 0.1           # Dropout 比率

# ============================================================================
# 训练配置
# ============================================================================

BATCH_SIZE = 64          # 每批样本数
MAX_SEQ_LEN = 512       # 最大序列长度（上下文长度）
LEARNING_RATE = 3e-4     # 最大学习率
MIN_LR = 3e-5            # 最小学习率 (LEARNING_RATE 的 1/10)
WARMUP_ITERS = 200       # 预热迭代次数（10%）
TRAIN_ITERS = 2000       # 训练迭代次数
EVAL_ITERS = 200         # 评估时的迭代次数
EVAL_INTERVAL = 200      # 每隔多少步评估一次

# 训练稳定性
WEIGHT_DECAY = 0.1       # 权重衰减（AdamW 正则化）
GRAD_CLIP = 1.0          # 梯度裁剪阈值

# DataLoader 配置
NUM_WORKERS = 4          # 数据加载进程数
PIN_MEMORY = True        # 锁页内存，加速 GPU 传输
PREFETCH_FACTOR = 2      # 每个 worker 预取的 batch 数

# 编译加速
USE_COMPILE = True       # 开启 torch.compile 加速（PyTorch 2.0+）

# ============================================================================
# Checkpoint 配置
# ============================================================================

CHECKPOINT_DIR = 'checkpoints'
SAVE_INTERVAL = 1000     # 每 1000 步保存一次
SAVE_BEST = True         # 保存最佳模型
RESUME = True            # 是否尝试从 checkpoint 恢复

# ============================================================================
# Swanlab 配置（可视化实验跟踪）
# ============================================================================

SWANLAB_PROJECT = 'nanogpt-lyrics'
SWANLAB_RUN_NAME = None  # None 则自动生成
USE_SWANLAB = True       # 是否启用 swanlab

# ============================================================================
# 生成配置
# ============================================================================

INITIAL_GENERATE_TOKENS = 100   # 训练前生成的 token 数
FINAL_GENERATE_TOKENS = 300     # 训练后生成的 token 数
GENERATE_PROMPT = "今夜我"       # 默认生成提示词

# 推理采样参数
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2

# ============================================================================
# MoE 配置
# ============================================================================

USE_MOE = False               # 是否启用 MoE（False 则使用原始 FeedForward）
NUM_EXPERTS = 8              # 路由专家数量
NUM_SHARED_EXPERTS = 1       # 共享专家数量（0=不使用共享专家）
TOP_K = 1                    # 每个 token 激活的路由专家数
MOE_FREQ = 2                 # 每隔几层使用 MoE（1=全部，2=交替）
AUX_LOSS_COEF = 0.01         # 负载均衡辅助损失系数

# ============================================================================
# YaRN 位置编码扩展配置
# ============================================================================

ROPE_SCALING_TYPE = 'none'    # 'none' = 标准 RoPE, 'yarn' = YaRN 动态扩展
ROPE_SCALING_FACTOR = 4.0     # 扩展倍数（例如 4.0 表示 4 倍长度扩展）
YARN_BETA_FAST = 32           # 高频阈值（波长短于此的维度保持原始）
YARN_BETA_SLOW = 1            # 低频阈值（波长长于此的维度完全插值）
YARN_ORIGINAL_MAX_SEQ_LEN = MAX_SEQ_LEN  # 原始训练长度（用于计算波长比例）

# ============================================================================
# 设备选择
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
