from lib.agent import Agent
from lib.tools import Tools
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import time

def evaluate(model_name="gpt-4o", experiment_tag="zero-shot",language = "en", local_version = False, sample = None):
    tools = Tools()

    if language == "en":
        if local_version:
            df = pd.read_parquet("hyr_ocr_process/output_parquet_hyr/EnglishOCR.parquet") # comment out if using TheFinAI/MultiFinBen-EnglishOCR
        else:
            ds_en = load_dataset("TheFinAI/OCR_Task", data_files = ["OCR_DATA/base64_encoded_version/EnglishOCR_3000_000.parquet"]) # use this line only if using TheFinAI/MultiFinBen-EnglishOCR 
            df = ds_en['train'].to_pandas() # use this line only if using TheFinAI/MultiFinBen-EnglishOCR
    elif language == "es":
        if local_version:
            df = pd.read_parquet("hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet")
        else:
            ds_es = load_dataset("TheFinAI/OCR_Task", data_files = ["OCR_DATA/base64_encoded_version/SpanishOCR_3000_000.parquet"])
            df = ds_es['train'].to_pandas()
    elif language == "gr":
        if local_version:
            df = pd.read_parquet("hyr_ocr_process/greek_output_parquet/GreekOCR_v1.parquet") #GreekOCR_500 have same structure as TheFinAI/MultiFinBen-EnglishOCR; GreekOCR_v1 has same structure as  output_parquet_hyr/EnglishOCR.parquet
        else:
            ds_gr = load_dataset("TheFinAI/OCR_Task", data_files = ["OCR_DATA/base64_encoded_version/GreekOCR_full_000.parquet"])
            df = ds_gr['train'].to_pandas()
    elif language == "jp":
        if local_version:
            df = pd.read_parquet("hyr_ocr_process/japanese_output_parquet/japanese_batch_0000.parquet")
        else:
            ds_jp = load_dataset("TheFinAI/OCR_Task", data_files = ["OCR_DATA/base64_encoded_version/JapaneseOCR_full_000.parquet"])
            df = ds_jp['train'].to_pandas()
    else: 
        print("Not a valid choice of language, please try again.")
        return language
    
    experiment_name = f"{model_name}_{experiment_tag}_financial"

    if language == "en":
        experiment_folder = os.path.join("hyr_results/predictions/", experiment_name)
    elif language == "es":
        experiment_folder = os.path.join("hyr_results/predictions_spanish/", experiment_name)
    elif language == "gr":
        experiment_folder = os.path.join("hyr_results/predictions_greek/", experiment_name)
    elif language == "jp":
        experiment_folder = os.path.join("hyr_results/predictions_japanese/", experiment_name)

    os.makedirs(experiment_folder, exist_ok=True)

    # Get predicted indices from filenames
    predicted_indices = set()
    if os.path.exists(experiment_folder):
        for fname in os.listdir(experiment_folder):
            if fname.startswith("pred_") and fname.endswith(".txt"):
                try:
                    idx = int(fname.replace("pred_", "").replace(".txt", ""))
                    predicted_indices.add(idx)
                except:
                    continue

    # Filter out completed predictions
    df = df[~df.index.isin(predicted_indices)]

    # Apply sample AFTER filtering
    if sample:
        df = df.head(sample)  # get sample

    agent = Agent(model_name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        # image_path = row["image_path"] # corresponds to local version
        image_path = row["image"] # corresponds to TheFinAI/MultiFinBen-EnglishOCR, image is in base64 format
        # ground_truth = row["matched_html"]
        output_file = os.path.join(experiment_folder, f"pred_{i}.txt")

        try:
            result = agent.draft(image_path)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            time.sleep(1.5)
            print(f"Finished processing")
        except Exception as e:
            print(f"⚠️ Error on index {i}: {e}")
            continue
        
    # for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
    #     image_path = row.get("image_path", row.get("image"))
    #     #image_path = os.path.join(local_dir, image_path).replace("./", "").replace("Japanese/", "")
    #     output_file = os.path.join(experiment_folder, f"pred_{i}.txt")

    #     try:
    #         result = agent.draft(image_path, local_version=local_version)
    #         with open(output_file, "w", encoding="utf-8") as f:
    #             f.write(result)
    #         # time.sleep(1)
    #     except Exception as e:
    #         print(f"⚠️ Error on index {i}: {e}")
    #         continue

def main():
    evaluate(model_name="Qwen/Qwen2.5-Omni-7B",language = "jp" , sample = 10)

if __name__ == '__main__':
    main()
