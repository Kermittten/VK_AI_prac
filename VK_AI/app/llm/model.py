import torch
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output"
BASE_MODEL_ID = "deepvk/llava-gemma-2b-lora"


def load_model():
    print("Загрузка базовой модели и адаптированных весов")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    base_model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Подгружаем обученные LoRA веса
    if OUTPUT_DIR.exists():
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        print("Веса успешно загружены из output")
    else:
        model = base_model
        print("Внимание: output не найден, используется базовая модель")

    model.to(device)
    model.eval()

    return model, processor