import torch
from PIL import Image
from llm.prompts import SYSTEM_PROMPT


def generate_answer(model, processor, image_path: str):
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer

    image = Image.open(image_path).convert("RGB").resize((336, 336))

    messages = [
        {"role": "user", "content": f"<image>\n{SYSTEM_PROMPT}"}
    ]
    
    prompt_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    NUM_IMAGE_TOKENS = 576
    image_token = "<image>"
    
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    expanded_prompt_ids = []
    for token_id in prompt_ids:
        if token_id == image_token_id:
            expanded_prompt_ids.extend([image_token_id] * NUM_IMAGE_TOKENS)
        else:
            expanded_prompt_ids.append(token_id)

    input_ids = torch.tensor([expanded_prompt_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Обработка изображения
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device, dtype=torch.float16)

    # Генерация ответа
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return raw_answer, raw_answer