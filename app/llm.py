"""
Module LLM - Tải và suy luận Qwen2.5-3B-Instruct cục bộ.
"""
import logging
import time
from threading import Lock

import torch

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_load_lock = Lock()
_device = None


def get_device() -> str:
    """Xác định thiết bị tính toán phù hợp."""
    global _device
    if _device is not None:
        return _device

    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_mb:.0f} MB")
        # Qwen2.5-3B cần ~6-7GB FP16; 4GB VRAM → dùng quantization hoặc CPU
        if vram_mb >= 6000:
            _device = "cuda"
        else:
            logger.info("VRAM không đủ cho FP16, sẽ thử load với quantization 4-bit lên GPU.")
            _device = "cuda_quantized"
    else:
        _device = "cpu"

    logger.info(f"Thiết bị suy luận: {_device}")
    return _device


def load_model():
    """Tải mô hình Qwen2.5-3B-Instruct từ đường dẫn local."""
    global _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = settings.LLM_MODEL_PATH
        device = get_device()
        logger.info(f"Đang tải Qwen2.5-3B-Instruct từ: {model_path}")

        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        start_time = time.time()

        if torch.cuda.is_available():
            logger.info("Sử dụng GPU NVIDIA (FP16) kết hợp CPU offload...")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Tối đa hóa VRAM sử dụng (GPU RTX 3050 Ti có 4096MB)
            max_memory = {0: "3450MB", "cpu": "16GB"}
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            logger.info("Sử dụng CPU inference với torch.float32...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True,
            )

        _model.eval()
        elapsed = time.time() - start_time
        logger.info(f"Đã tải Qwen thành công trong {elapsed:.1f}s. Device: {_model.device}")

        # Kiểm tra model thực sự chạy trên CPU hay GPU
        logger.info(f"Device map thực tế: {getattr(_model, 'hf_device_map', 'Không có')}")

        if torch.cuda.is_available():
            logger.info(
                f"VRAM đang cấp phát: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            )


def generate(messages: list[dict], max_new_tokens: int = None) -> str:
    """
    Sinh câu trả lời từ Qwen2.5-3B-Instruct.

    Args:
        messages: Danh sách tin nhắn theo format chat
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_new_tokens: Số token tối đa được sinh.

    Returns:
        Câu trả lời dạng text.

    """
    if _model is None or _tokenizer is None:
        load_model()

    if max_new_tokens is None:
        max_new_tokens = settings.MAX_NEW_TOKENS

    try:
        start_time = time.time()

        # Áp dụng chat template của Qwen
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = _tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Lấy phần response (bỏ phần input)
        input_length = inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_length:]
        response = _tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Loại bỏ các ký tự liệt kê dở dang ở cuối như "2.", "3.", "-"
        lines = [line.strip() for line in response.split("\n")]
        while lines and (lines[-1] in ("1.", "2.", "3.", "4.", "5.", "-", "*", "•") or len(lines[-1]) <= 2):
            lines.pop()
        response = "\n".join(lines).strip()

        # Đảm bảo câu trả lời luôn trọn vẹn, không bị cắt xén oan các ý chính
        if response and response[-1] not in (".", "!", "?", "”", '"', "…"):
            last_punct = max(response.rfind("."), response.rfind("!"), response.rfind("?"))
            # Chỉ cắt bớt nếu phần sau dấu chấm là mẩu vụn cụt ngủn (< 25 ký tự)
            if last_punct > 30 and (len(response) - last_punct) < 25:
                response = response[:last_punct + 1].strip()
            else:
                response = response + "."

        elapsed = time.time() - start_time
        token_count = len(response_ids)
        logger.info(f"LLM response: {token_count} tokens trong {elapsed:.1f}s ({token_count/elapsed:.1f} tok/s)")

        return response

    except Exception as e:
        logger.error(f"Lỗi sinh câu trả lời: {e}")
        return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."


def is_model_loaded() -> bool:
    """Kiểm tra model đã được tải chưa."""
    return _model is not None and _tokenizer is not None
