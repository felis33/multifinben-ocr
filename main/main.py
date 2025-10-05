from lib.agent import Agent
from lib.tools import Tools
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from urllib.parse import quote, unquote
import time


LANGUAGE_OUTPUT_ROOT = {
    "en": Path("hyr_results/predictions"),
    "es": Path("hyr_results/predictions_spanish"),
    "gr": Path("hyr_results/predictions_greek"),
    "jp": Path("hyr_results/predictions_japanese"),
}


def _build_experiment_folder(language: str, model_name: str, experiment_tag: str) -> Path:
    try:
        base_dir = LANGUAGE_OUTPUT_ROOT[language]
    except KeyError:
        raise ValueError(f"Unsupported language '{language}'.")

    safe_model_name = model_name.replace("/", "__")
    experiment_name = f"{safe_model_name}_{experiment_tag}_financial"
    experiment_folder = base_dir / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder


def _collect_completed_ids(experiment_folder: Path) -> set[str]:
    completed = set()
    if not experiment_folder.exists():
        return completed

    for file_path in experiment_folder.iterdir():
        if not file_path.is_file():
            continue
        stem = file_path.stem
        if not stem.startswith("pred_"):
            continue
        encoded_id = stem[len("pred_"):]
        completed.add(unquote(encoded_id))
    return completed


def _encode_sample_id(sample_id: str) -> str:
    encoded = quote(sample_id, safe="")
    return encoded or "sample"

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
            df = df.iloc[:600]
            #df = df.iloc[-600:]
            
    else: 
        print("Not a valid choice of language, please try again.")
        return language
    
    experiment_folder = _build_experiment_folder(language, model_name, experiment_tag)

    # Get predicted indices from filenames
    predicted_ids = _collect_completed_ids(experiment_folder)

    # Filter out completed predictions
    df = df[~df.index.astype(str).isin(predicted_ids)]

    # Apply sample AFTER filtering
    if sample:
        df = df.head(sample)  # get sample

    agent = Agent(model_name)
    

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        # image_path = row["image_path"] # corresponds to local version
        image_path = row["image"] # corresponds to TheFinAI/MultiFinBen-EnglishOCR, image is in base64 format
        # ground_truth = row["matched_html"]
        sample_id = str(i)
        encoded_sample_id = _encode_sample_id(sample_id)
        output_file = experiment_folder / f"pred_{encoded_sample_id}"

        try:
            result = agent.draft(image_path)
            print(result)
            tools.save_text(result, output_file, suffix=".html")
            time.sleep(1)
            print(f"Finished processing")
            
        except Exception as e:
            print(f"⚠️ Error on index {i}: {e}")
            continue
    
def main():
    evaluate(model_name="Qwen/Qwen2.5-Omni-7B",language = "jp" , sample = 500)

if __name__ == '__main__':
    main()
