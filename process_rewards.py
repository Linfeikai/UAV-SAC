import os
import re
import pandas as pd
from collections import defaultdict
import argparse  # 导入 argparse 模块


def add_statistics_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    为一个DataFrame计算统计数据（和、平均值）并附加到末尾。
    """
    if df.empty or len(df.columns) < 2:
        return df

    cols_to_calculate = df.columns[1:]
    sums = df[cols_to_calculate].sum()
    means = df[cols_to_calculate].mean()
    grand_total = sums.sum()

    df.loc["spacer"] = [None] * len(df.columns)

    sum_row_label = "总和 (Sum)"
    df.loc[sum_row_label] = [None] * len(df.columns)
    df.loc[sum_row_label, df.columns[0]] = sum_row_label
    df.loc[sum_row_label, cols_to_calculate] = sums

    mean_row_label = "平均值 (Mean)"
    df.loc[mean_row_label] = [None] * len(df.columns)
    df.loc[mean_row_label, df.columns[0]] = mean_row_label
    df.loc[mean_row_label, cols_to_calculate] = means

    grand_total_col_name = "总计 (Grand Total)"
    df[grand_total_col_name] = None
    df.loc[sum_row_label, grand_total_col_name] = grand_total

    return df


def process_reward_files(file_prefix: str):  # <--- 函数现在接收一个前缀参数
    """
    查找、合并、统计并删除 reward CSV 文件。
    """
    file_pairs = defaultdict(dict)
    pattern = re.compile(r"^(unweighted_rewards|weighted_rewards)_(\d+)\.csv$")

    print("--- 开始扫描文件 ---")
    current_directory = os.getcwd()
    for filename in os.listdir(current_directory):
        match = pattern.match(filename)
        if match:
            prefix_type = match.group(1).split("_")[0]
            timestamp = match.group(2)
            file_pairs[timestamp][prefix_type] = filename

    if not file_pairs:
        print("在当前目录下没有找到任何匹配的 reward 文件。")
        return

    print(f"\n--- 找到了 {len(file_pairs)} 组可能的文件对，开始处理 ---")
    processed_count = 0
    for timestamp, files in file_pairs.items():
        if "unweighted" in files and "weighted" in files:
            unweighted_csv = files["unweighted"]
            weighted_csv = files["weighted"]

            # --- 【核心改动】根据传入的前缀参数构造输出文件名 ---
            if file_prefix:
                # 如果前缀不为空，则使用 "前缀_rewards_时间戳.xlsx" 格式
                output_excel = f"{file_prefix}_rewards_{timestamp}.xlsx"
            else:
                # 如果没有提供前缀，则保持原来的格式
                output_excel = f"rewards_{timestamp}.xlsx"
            # ----------------------------------------------------

            print(f"\n正在处理时间戳: {timestamp}")
            print(f"  -> 输入文件1: {unweighted_csv}")
            print(f"  -> 输入文件2: {weighted_csv}")
            print(f"  -> 输出文件: {output_excel}")

            try:
                df_unweighted = pd.read_csv(unweighted_csv)
                df_weighted = pd.read_csv(weighted_csv)

                df_unweighted_processed = add_statistics_to_dataframe(df_unweighted)
                df_weighted_processed = add_statistics_to_dataframe(df_weighted)

                with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
                    df_unweighted_processed.to_excel(
                        writer, sheet_name="unweighted", index=False
                    )
                    df_weighted_processed.to_excel(
                        writer, sheet_name="weighted", index=False
                    )

                print(f"  [成功] 已创建带统计的 Excel 文件: {output_excel}")

                os.remove(unweighted_csv)
                os.remove(weighted_csv)
                print(f"  [成功] 已删除原始 CSV 文件。")
                processed_count += 1

            except Exception as e:
                print(f"  [失败] 处理时间戳 {timestamp} 时发生错误: {e}")
                print(f"  将保留原始文件，请手动检查。")
        else:
            print(f"\n警告: 时间戳 {timestamp} 的文件不完整，已跳过。")
            print(f"  找到的文件: {list(files.values())}")

    print(f"\n--- 全部处理完毕 ---")
    print(f"总共成功处理了 {processed_count} 对文件。")


# 运行主函数
if __name__ == "__main__":
    # --- 【新增】设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="合并 rewards CSV 文件到 Excel, 计算统计数据, 并提供可选的文件名前缀。"
    )
    parser.add_argument(
        "-p",
        "--prefix",  # 参数的名称，-p是简写，--prefix是全写
        type=str,  # 参数的类型是字符串
        default="",  # 如果不提供此参数，默认值为空字符串
        help="为所有输出的 Excel 文件添加一个统一的前缀, 例如: --prefix Experiment1",
    )
    args = parser.parse_args()  # 解析命令行传入的参数
    # ------------------------------------

    # 将解析到的前缀传递给主函数
    process_reward_files(file_prefix=args.prefix)
