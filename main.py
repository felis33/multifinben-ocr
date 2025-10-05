from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from rich import print
from rich.progress import track

# Global configuration
FOLDER_PATH = "data"  # Root folder containing subfolders with images
TEST_RUN = False  # If True, only process first 5 images for testing

# Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")


def process_image_to_html(image_path: str) -> str:
    """
    Convert a financial statement image to HTML using Qwen2.5-Omni model.

    Args:
        image_path: Path to the image file

    Returns:
        Generated HTML text
    """
    prompt = "Convert this financial statement page into semantically correct HTML. Return html and nothing else. Use plain html only, no styling please."

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
            ],
        },
    ]

    # set use audio in video
    USE_AUDIO_IN_VIDEO = False

    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(  # type: ignore
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",  # type: ignore
        padding=True,  # type: ignore
        use_audio_in_video=USE_AUDIO_IN_VIDEO,  # type: ignore
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Store the length of input tokens
    input_length = inputs['input_ids'].shape[1]

    # Inference: Generation of the output text and audio
    text_ids, _ = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # Slice to get only the newly generated tokens
    generated_ids = text_ids[:, input_length:]

    # Decode only the generated tokens
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def process_batch():
    """
    Process all images in subfolders and save results as text files.

    Expects structure:
    FOLDER_PATH/
        subfolder1/
            image.png (or .jpg, .jpeg)
        subfolder2/
            image.png
        ...

    Outputs:
    FOLDER_PATH/
        subfolder1.txt
        subfolder2.txt
        ...
    """
    root_path = Path(FOLDER_PATH)

    if not root_path.exists():
        print(f"[red]Error: Folder path '{FOLDER_PATH}' does not exist[/red]")
        return

    # Get all subfolders
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]

    if not subfolders:
        print(f"[yellow]No subfolders found in '{FOLDER_PATH}'[/yellow]")
        return

    # Limit to 5 for test run
    if TEST_RUN:
        subfolders = subfolders[:5]
        print(f"[cyan]Test mode: Processing only {len(subfolders)} folders[/cyan]")
    else:
        print(f"[green]Found {len(subfolders)} folders to process[/green]")

    # Process each subfolder with progress bar
    for subfolder in track(subfolders, description="Processing images..."):
        # Find image in subfolder (support png, jpg, jpeg)
        image_files = list(subfolder.glob("*.png")) + \
                     list(subfolder.glob("*.jpg")) + \
                     list(subfolder.glob("*.jpeg"))

        if not image_files:
            print(f"[yellow]Warning: No image found in {subfolder.name}[/yellow]")
            continue

        # Use the first image found
        image_path = str(image_files[0])

        try:
            # Process the image
            result = process_image_to_html(image_path)

            # Write result to text file with folder name
            output_file = root_path / f"{subfolder.name}.txt"
            output_file.write_text(result, encoding="utf-8")

        except Exception as e:
            print(f"[red]Error processing {subfolder.name}: {e}[/red]")

    print("[green]✓ Batch processing complete![/green]")


if __name__ == "__main__":
    process_batch()