from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="TheFinAI/OCR_Task",
    repo_type="dataset",
    local_dir="dataset",
    revision="main",
    allow_patterns="japanese_images/*",  # Only download japanese_images folder
    ignore_patterns=None,
    max_workers=4,
    local_dir_use_symlinks=False
)

print("saved to:", local_path)