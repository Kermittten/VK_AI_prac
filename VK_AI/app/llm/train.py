import os
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from llm.prompts import SYSTEM_PROMPT

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"


class ClothingDataset(Dataset):
    def __init__(self, json_path: Path):
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = DATASET_DIR / "images" / item["image"]
        
        image = Image.open(image_path).convert("RGB").resize((336, 336))
        answer = json.dumps(item["answer"], ensure_ascii=False)
        return image, answer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "deepvk/llava-gemma-2b-lora"

    print("Загрузка модели и процессора...")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    annotations_path = DATASET_DIR / "annotations.json"
    dataset = ClothingDataset(annotations_path)

    NUM_IMAGE_TOKENS = 576
    
    image_token = "<image>"

    def collate_fn(batch):
        images, answers = zip(*batch)
        
        input_ids_list = []
        labels_list = []

        for img, ans in zip(images, answers):
            messages = [
                {"role": "user", "content": f"<image>\n{SYSTEM_PROMPT}"}
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            
            expanded_prompt_ids = []
            for token_id in prompt_ids:
                if token_id == image_token_id:
                    expanded_prompt_ids.extend([image_token_id] * NUM_IMAGE_TOKENS)
                else:
                    expanded_prompt_ids.append(token_id)

            target_ids = tokenizer(f"{ans}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]

            input_ids = expanded_prompt_ids + target_ids
            
            labels = [-100] * len(expanded_prompt_ids) + target_ids

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Паддинг батча
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids_padded.ne(pad_id).long()

        # Обработка картинок
        image_inputs = processor.image_processor(images=list(images), return_tensors="pt")

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "pixel_values": image_inputs["pixel_values"].to(torch.float16),
            "labels": labels_padded
        }

    train_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    epochs = 5
    accum_steps = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    total_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=max(1, total_steps)
    )

    print("Начало обучения")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps

            loss.backward()

            current_loss_val = loss.item() * accum_steps
            total_loss += current_loss_val

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 5 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch + 1}/{epochs}] | Step [{step + 1}/{len(train_loader)}] | Loss: {current_loss_val:.4f}")

            del batch, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch + 1} завершена. Средний Loss: {avg_loss:.4f}\n")

    print(f"Сохранение модели в {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Конец")


if __name__ == "__main__":
    train()