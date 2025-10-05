import os
import pandas as pd
from tqdm import tqdm
from evaluate import load
from lib.tools import Tools

rouge = load("rouge")

def evaluate_rouge(pred_dir, ground_truths, model_name="gpt-4o",lang = 'en'):
    records = []

    for i in ground_truths.index:
        gt = ground_truths.loc[i]
        if pd.isna(gt) or not isinstance(gt, str):
            continue

        pred_path = os.path.join(pred_dir, f"{model_name}_pred_{i}.txt")
        if not os.path.exists(pred_path):
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = f.read().strip()
            clean_pred = pred
            # if lang != "es":
            #     import re
            #     clean_pred = re.sub(r"<[^>]+>", " ", pred)
            #     clean_pred = re.sub(r"\s+", " ", clean_pred).strip()


        try:
            rouge_score = rouge.compute(predictions=[clean_pred], references=[gt], use_stemmer=True)
            rouge_1_f1 = float(rouge_score["rouge1"])
        except Exception as e:
            print(f"ROUGE error on index {i}: {e}")
            rouge_1_f1 = None

        records.append({
            "index": i,
            "ground_truth": gt,
            "prediction": pred,
            "ROUGE-1": rouge_1_f1,
            "Model": model_name
        })

    return pd.DataFrame(records)

def run_rouge_eval(parquet_path, pred_dir, model_name="gpt-4o",lang = "en",output_csv = None):
    df = pd.read_parquet(parquet_path)

    # Extract indices from prediction files
    pred_indexes = []
    for fname in os.listdir(pred_dir):
        if fname.startswith(f"{model_name}_pred_") and fname.endswith(".txt"):
            try:
                idx = int(fname.replace(f"{model_name}_pred_", "").replace(".txt", ""))
                pred_indexes.append(idx)
            except:
                continue

    df = df.loc[df.index.intersection(pred_indexes)]
    df_eval = evaluate_rouge(pred_dir, df["matched_html"], model_name=model_name,lang = lang)

    if output_csv:
        os.makedirs("hyr_results/eval_rouge", exist_ok=True)
        df_eval.to_csv(output_csv, index=False)
        print(f"✅ Evaluation saved to CSV: {output_csv}")
    return df_eval

def main():
    run_rouge_eval(
        # parquet_path="hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet",
        # parquet_path="hyr_ocr_process/output_parquet_hyr/EnglishOCR.parquet",
        # pred_dir="hyr_results/predictions_spanish/gpt-4o_zero-shot_financial",
        # pred_dir="hyr_results/predictions/gpt-5_zero-shot_financial",
        # model_name="gpt-4o",
        # lang = "es",
        # lang = "en",
        # output_csv="hyr_results/eval_rouge/eval_gpt_4o_spanish_rouge.csv"
        # output_csv="hyr_results/eval_rouge/eval_gpt_5_rouge.csv"

        parquet_path="hyr_ocr_process/japanese_output_parquet/japanese_batch_0000.parquet",
        pred_dir="hyr_results/predictions_japanese/gpt-4o_zero-shot_financial",
        model_name="gpt-4o",
        lang = "jp",
        output_csv="hyr_results/eval_rouge/eval_japanese_gpt_4o_rouge.csv"

        # parquet_path="hyr_ocr_process/greek_output_parquet/GreekOCR_v1.parquet",
        # pred_dir="hyr_results/predictions_greek/gpt-4o_zero-shot_financial",
        # model_name="gpt-4o",
        # lang = "gr",
        # output_csv="hyr_results/eval_rouge/eval_greek_gpt_4o_rouge.csv"
    )

if __name__ == '__main__':

    main()
