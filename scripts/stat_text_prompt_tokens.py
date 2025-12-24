import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# 1. 配置参数
MODEL_PATH = '/share_data/data1/models/Qwen/Qwen3-4B-Base'  # 替换为你实际的本地模型路径或 HuggingFace ID
DATA_PATH = '/share_data/data1/fanshengda/DEvo/data/solver_1221/solver_questioner_350_train.parquet'
SAVE_PLOT_NAME = 'token_distribution_analysis.png'

# 设置绘图风格
plt.switch_backend('Agg')  # 服务器环境建议使用 Agg 后端，直接保存图片
sns.set_theme(style="whitegrid")
tqdm.pandas()


def main():
    # 2. 加载数据
    print(f"正在加载数据: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # 3. 初始化 Tokenizer
    print(f"正在加载 Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 4. 定义统计函数
    def count_tokens(chat_list):
        try:
            # 使用 apply_chat_template 还原真实输入场景
            # add_generation_prompt=False 表示统计纯 Prompt 长度
            # tokenize=True 直接返回 token ids 列表
            tokens = tokenizer.apply_chat_template(
                chat_list,
                tokenize=True,
                add_generation_prompt=False,
                extra_turn_indicators=False  # 部分模型模板需要，根据需求开启
            )
            return len(tokens)
        except Exception as e:
            # 容错处理：部分数据可能格式不规范
            return 0

    # 5. 执行统计
    cols_to_stat = ['prompt', 'text_prompt']
    for col in cols_to_stat:
        if col in df.columns:
            print(f"正在处理列: {col} ...")
            df[f'{col}_len'] = df[col].progress_apply(count_tokens)
        else:
            print(f"警告: 列 {col} 不存在于数据中")

    # 6. 打印统计报告 (P95, P99 对 RL 调参至关重要)
    print("\n" + "=" * 30)
    print("Token 长度统计报告")
    print("=" * 30)
    stats = df[[f'{col}_len' for col in cols_to_stat if col in df.columns]].describe(
        percentiles=[.5, .75, .9, .95, .99]
    )
    print(stats)
    print("=" * 30)

    # 7. 绘图
    print("正在生成分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图 A: 直方图 + KDE 分布
    for col in cols_to_stat:
        len_col = f'{col}_len'
        if len_col in df.columns:
            sns.histplot(df[len_col], kde=True, ax=axes[0], label=col, element="step", alpha=0.4)

    axes[0].set_title('Token Length Density (Histogram & KDE)')
    axes[0].set_xlabel('Number of Tokens')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # 图 B: 累计分布函数 (CDF) - 用于确定截断长度 (Max Seq Len)
    for col in cols_to_stat:
        len_col = f'{col}_len'
        if len_col in df.columns:
            sns.ecdfplot(data=df[len_col], ax=axes[1], label=f'{col} (CDF)')

    # 在 95% 处画一条参考线
    axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.6, label='95th Percentile')
    axes[1].set_title('Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Number of Tokens')
    axes[1].set_ylabel('Proportion of Samples')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(SAVE_PLOT_NAME, dpi=300)
    print(f"分析图表已保存至: {SAVE_PLOT_NAME}")


if __name__ == "__main__":
    main()