"""One resident vision-language model that does everything: image transcription,
query expansion, and answering. Loaded once, by the server process only."""
from __future__ import annotations

import importlib
import logging
import os

# It gets or creates a logger named mc3.llm
log = logging.getLogger("mc3.llm")

MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/vlm")


class HFModel:
    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            log.error("NO GPU VISIBLE - running on CPU. The harness rejects a run that never touches VRAM.")
        px = 28 * 28
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_dir, min_pixels=256 * px, max_pixels=1600 * px)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_dir)
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except TypeError:  # older transformers spell it torch_dtype
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.model = model.to(self.device).eval()
        log.info("model loaded on %s", self.device)

    def generate(self, prompt: str, image=None, max_new_tokens: int = 128, tag: str = "") -> str:
        torch = self.torch
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image] if image is not None else None,
                                return_tensors="pt")
        inputs = {k: (v.to(self.device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


def make_llm():
    """MC3_LLM_FACTORY='module:Class' swaps in a stand-in (used only by the local tests)."""
    spec = os.environ.get("MC3_LLM_FACTORY")
    if spec:
        mod, attr = spec.split(":")
        return getattr(importlib.import_module(mod), attr)()
    return HFModel()
