import pandas as pd
import zipfile
import json
import numpy as np

SUBMISSION_ZIP = "submission.zip"
GT_PATH = "ground_truth_labels.csv"


def load_predictions_from_zip(zip_path):
    """
    从选手提交的 zip 中加载预测结果。
    假设 submissionA.csv 和 submissionB.csv 均无表头，
    每一行包含两个数值（第一列 KD 数量，第二列 SD 数量）。
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("submissionA.csv") as f:
            # .values 直接将 DataFrame 转换为 NumPy 矩阵 (N行 x 2列)
            pred_A = pd.read_csv(f, header=None).values
        with z.open("submissionB.csv") as f:
            pred_B = pd.read_csv(f, header=None).values
    return pred_A, pred_B


def load_ground_truth(gt_path):
    """
    加载官方的 Ground Truth 文件。
    假设 ground_truth_labels.csv 带有表头，至少包含三列：
    'subset' (值为 'A' 或 'B'), 'label_kd', 'label_sd'
    """
    df = pd.read_csv(gt_path)
    labels_A = df[df["subset"] == "A"][["label_kd", "label_sd"]].values
    labels_B = df[df["subset"] == "B"][["label_kd", "label_sd"]].values
    return labels_A, labels_B


def evaluate(preds, labels, penalty_factor=0.1):
    """
    计算双乐器计数任务的 MAE，并转化为百分制分数。
    公式: Score = max(0, 1 - penalty_factor * MAE_total)
    """
    if len(preds) != len(labels):
        print(f"Length mismatch: preds={len(preds)}, labels={len(labels)}")
        return 0.0

    preds = np.array(preds)
    labels = np.array(labels)
    
    if preds.shape[1] != 2 or labels.shape[1] != 2:
        print(f"Shape error: preds shape={preds.shape}, labels shape={labels.shape}")
        return 0.0

    abs_errors = np.abs(preds - labels)
    
    mae_per_sample = np.mean(abs_errors, axis=1)
    total_mae = np.mean(mae_per_sample)
    
    score = max(0.0, 1.0 - penalty_factor * total_mae)
    
    return float(score)


def save_score_json(score_a, score_b):
    """
    将计算出的分数按照原有格式写入 score.json
    """
    result = {
        "status": True,
        "score": {"public_a": round(score_a, 4), "private_b": round(score_b, 4)},
        "msg": "MAE scoring completed.",
    }
    with open("score.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Score written to score.json")


if __name__ == "__main__":
    preds_a, preds_b = load_predictions_from_zip(SUBMISSION_ZIP)
    labels_a, labels_b = load_ground_truth(GT_PATH)
    print

    score_a = evaluate(preds_a, labels_a)
    score_b = evaluate(preds_b, labels_b)

    save_score_json(score_a, score_b)