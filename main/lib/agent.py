from transformers import AutoProcessor, AutoModelForVision2Seq, BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from PIL import Image, UnidentifiedImageError
import torch
from openai import OpenAI
import base64
import io
import os
import binascii

torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "llava" in model_name:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                quantization_config=bnb_config,
            ).eval()

        elif "finllava" in model_name.lower():
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.processor = AutoProcessor.from_pretrained(
                "TheFinAI/FinLLaVA", trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                "TheFinAI/FinLLaVA",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                quantization_config=bnb_config,
            ).eval()

        elif "blip" in model_name:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base", trust_remote_code=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()

        elif "qwen" in model_name.lower():

            if "omni" in model_name.lower():
                self.processor = Qwen2_5OmniProcessor.from_pretrained(
                    "Qwen/Qwen2.5-Omni-7B", trust_remote_code=True
                )
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-Omni-7B",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    ),
                ).eval()

            else:
                self.processor = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-VL-Max", trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-VL-Max",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    ),
                ).eval()

        elif "deepseek" in model_name:
            model_path = "deepseek-ai/deepseek-vl-7b-chat"

            self.processor = VLChatProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer

            self.model = MultiModalityCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()

        elif "llama" in model_name.lower():
            model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="flex_attention",
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()

        elif "gemma" in model_name:

            if "4b" in model_name:
                model_id = "google/gemma-3-4b-it"
            else:
                model_id = "google/gemma-3-27b-it"

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()
        elif "gpt-4o" in self.model_name or "o3-mini" in self.model_name or \
            "gpt-5" in self.model_name:
            self.openai_api_key = XXXXXXXXXXXXXXXXXXXXXX  # Replace with your API key
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _load_image(self, source):
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, (bytes, bytearray)):
            return Image.open(io.BytesIO(source)).convert("RGB")

        if isinstance(source, dict):
            # Hugging Face datasets often store images as {"path": ..., "bytes": ...}
            if "bytes" in source:
                return Image.open(io.BytesIO(source["bytes"])).convert("RGB")
            if "path" in source and source["path"] and os.path.exists(source["path"]):
                return Image.open(source["path"]).convert("RGB")

        if isinstance(source, str):
            trimmed = source.strip()
            if os.path.exists(trimmed):
                return Image.open(trimmed).convert("RGB")

            # Handle base64 strings (optionally prefixed with data URI)
            if trimmed.startswith("data:"):
                _, _, trimmed = trimmed.partition(",")
            try:
                decoded = base64.b64decode("".join(trimmed.split()), validate=True)
                return Image.open(io.BytesIO(decoded)).convert("RGB")
            except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
                pass

        raise ValueError("Unsupported image input; expected file path, base64 string, bytes, dict, or PIL Image")

    def draft(self, image_input):
        image = self._load_image(image_input)
        
        prompt = "Convert this financial statement page into semantically correct HTML. Return html and nothing else. Use plain html only, no styling please."

        if "llava" in self.model_name.lower():
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(
                self.device
            )

            with torch.no_grad():
                output = self.model.generate(inputs, max_new_tokens=1024)

            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "blip" in self.model_name:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1024)
            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "qwen" in self.model_name.lower():

            if "omni" in self.model_name.lower():
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
                            {"type": "image", "image": image},
                        ],
                    },
                ]

                
                USE_AUDIO_IN_VIDEO = False

                # Preparation for inference
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                audios, images, videos = process_mm_info(  # type: ignore
                    conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
                )


                inputs = self.processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",  # type: ignore
                    padding=True,  # type: ignore
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,  # type: ignore
                )
                inputs = inputs.to(self.model.device).to(self.model.dtype)
                input_length = inputs['input_ids'].shape[1]
                
                text_ids, _ = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                
                generated_ids = text_ids[:, input_length:]

                # Decode only the generated tokens
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            else:
                query = self.processor.from_list_format(
                    [
                        {"image": image},
                        {"text": prompt},
                    ]
                )

                inputs = self.processor(query, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=1024)

                result = self.processor.decode(output[0], skip_special_tokens=True)
            
            print(output_text[0])
            return output_text[0]

        elif "deepseek" in self.model_name.lower():

            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{prompt}",
                    "images": [f"{image_path}"],
                },
                {"role": "Assistant", "content": ""},
            ]

            pil_images = load_pil_images(conversation)

            prepare_inputs = self.processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(
                self.model.device,
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            with torch.no_grad():

                # run image encoder to get the image embeddings
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

                # run the model to get the response
                output = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=True,
                )

            result = self.processor.tokenizer.decode(
                output[0].cpu().tolist(), skip_special_tokens=True
            )

            return result

        elif "llama" in self.model_name.lower():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(
                self.device,
                torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1024)

            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "gemma" in self.model_name:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(
                self.model.device,
                torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1024)

            decoded = processor.decode(generation, skip_special_tokens=True)

            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "gpt-4o" in self.model_name:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            
            client = OpenAI(
                # This is the default and can be omitted
                api_key=self.openai_api_key,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=2048
            )
            return response.choices[0].message.content
        
        elif "gpt-5" in self.model_name:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            
            client = OpenAI(
                # This is the default and can be omitted
                api_key=self.openai_api_key,
            )
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}",
                                                                "detail": "high"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_completion_tokens=256
                
            )
            res = response.choices[0].message.content
            print(res)
            return res
