from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def expand_paths(paths: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for path in paths:
        matches = glob.glob(path)
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(path)
    return expanded



def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                # Keep text-only fine-tuning simple. Ignore image blobs in the text-only path.
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif "content" in item:
                    chunks.append(str(item.get("content", "")))
        return "\n".join([c for c in chunks if c])
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        if "content" in content:
            return str(content.get("content", ""))
    return str(content)



def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())



def example_to_messages(example: dict[str, Any], *, for_generation: bool = False) -> list[dict[str, str]]:
    """Normalize common JSONL formats into chat messages.

    Accepted shapes:
    - {"messages": [...]}  # OpenAI-style
    - {"prompt": "...", "response": "..."}
    - {"instruction": "...", "input": "...", "output": "..."}
    - {"text": "..."}  # wrapped as a single user message

    For RL prompt-only generation, set for_generation=True to drop a final assistant turn.
    """
    if "messages" in example and example["messages"]:
        messages = [
            {"role": str(m["role"]), "content": _content_to_text(m.get("content", ""))}
            for m in example["messages"]
        ]
        if for_generation and messages and messages[-1]["role"] == "assistant":
            if messages[-1]["content"].strip():
                messages = messages[:-1]
        return messages

    if "prompt" in example:
        messages = [{"role": "user", "content": _content_to_text(example["prompt"])}]
        if not for_generation and "response" in example:
            messages.append({"role": "assistant", "content": _content_to_text(example.get("response", ""))})
        return messages

    if "instruction" in example or "output" in example:
        instruction = _content_to_text(example.get("instruction", ""))
        input_text = _content_to_text(example.get("input", ""))
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\n{input_text}" if instruction else input_text
        messages = [{"role": "user", "content": prompt}]
        if not for_generation and "output" in example:
            messages.append({"role": "assistant", "content": _content_to_text(example.get("output", ""))})
        return messages

    if "text" in example:
        return [{"role": "user", "content": _content_to_text(example["text"])}]

    raise ValueError(
        "Unsupported example format. Expected one of: messages, prompt/response, instruction/output, or text."
    )



def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {model_id!r} does not expose a chat template. "
            "Qwen3.5 should have one in a recent Transformers build."
        )
    return tokenizer



def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")



def load_text_only_causal_lm(
    model_id: str,
    *,
    dtype: str = "bf16",
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    attn_implementation: str | None = None,
):
    """Load the Qwen3.5 text-only LM path.

    The official Transformers docs expose Qwen3_5ForCausalLM for text-only use. In a
    recent enough Transformers build, AutoModelForCausalLM resolves that mapping.
    """
    model_kwargs: dict[str, Any] = {"torch_dtype": _resolve_dtype(dtype)}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception:
        # Fallback for slightly older AutoModel mappings.
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.config.use_cache = False if gradient_checkpointing else True
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if use_lora:
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model



def render_messages_as_prompt(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )



def render_midtrain_text(example: dict[str, Any], tokenizer) -> str:
    if "text" in example and example["text"]:
        return _content_to_text(example["text"])
    messages = example_to_messages(example, for_generation=False)
    return render_messages_as_prompt(tokenizer, messages, add_generation_prompt=False)



def build_midtrain_dataset(
    *,
    train_files: list[str],
    tokenizer,
    block_size: int,
    num_proc: int | None = None,
) -> Dataset:
    data_files = expand_paths(train_files)
    dataset = load_dataset("json", data_files=data_files, split="train")

    def render_row(example: dict[str, Any]) -> dict[str, str]:
        text = render_midtrain_text(example, tokenizer)
        if tokenizer.eos_token:
            text = text + tokenizer.eos_token
        return {"text": text}

    rendered = dataset.map(render_row, remove_columns=dataset.column_names)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = rendered.map(tokenize, batched=True, num_proc=num_proc, remove_columns=rendered.column_names)

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        result = {
            k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, v in concatenated.items()
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    grouped = tokenized.map(group_texts, batched=True, num_proc=num_proc)
    return grouped



def build_sft_features(example: dict[str, Any], tokenizer, max_length: int) -> dict[str, list[int]]:
    messages = example_to_messages(example, for_generation=False)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT examples must end with an assistant turn.")

    prompt_messages = messages[:-1]
    assistant_text = messages[-1]["content"]
    prompt_text = render_messages_as_prompt(tokenizer, prompt_messages, add_generation_prompt=True)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(assistant_text + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }



def build_sft_dataset(
    *,
    train_files: list[str],
    tokenizer,
    max_length: int,
    num_proc: int | None = None,
) -> Dataset:
    data_files = expand_paths(train_files)
    dataset = load_dataset("json", data_files=data_files, split="train")

    def to_features(example: dict[str, Any]) -> dict[str, list[int]]:
        return build_sft_features(example, tokenizer, max_length)

    return dataset.map(to_features, remove_columns=dataset.column_names, num_proc=num_proc)



def build_messages_dataset(train_files: list[str]) -> Dataset:
    """For TRL GKD/OPD, normalize any supported row format into a `messages` column."""
    data_files = expand_paths(train_files)
    dataset = load_dataset("json", data_files=data_files, split="train")

    def to_messages(example: dict[str, Any]) -> dict[str, Any]:
        messages = example_to_messages(example, for_generation=False)
        if not messages:
            raise ValueError("Could not build messages for example")
        # If the dataset is prompt-only, append an empty assistant stub. GKD will sample
        # on-policy outputs when lmbda=1.0, so the prompt scaffold is what matters.
        if messages[-1]["role"] != "assistant":
            messages = [*messages, {"role": "assistant", "content": ""}]
        return {"messages": messages}

    return dataset.map(to_messages, remove_columns=dataset.column_names)


@dataclass
class CausalPaddingCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        pad_id = self.tokenizer.pad_token_id
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feat in features:
            pad_len = max_len - len(feat["input_ids"])
            batch["input_ids"].append(feat["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feat["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feat["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}



def compute_completion_logprobs(
    model,
    sequences: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logprobs for completion tokens and a mask marking valid completion tokens.

    sequences shape: [batch, seq_len]
    prompt_lengths shape: [batch]
    """
    attention_mask = sequences.ne(pad_token_id).long()
    outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = sequences[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    seq_lens = attention_mask.sum(dim=-1)
    token_mask = torch.zeros_like(gathered, dtype=torch.float32)
    for i in range(sequences.size(0)):
        start = max(int(prompt_lengths[i].item()) - 1, 0)
        end = max(int(seq_lens[i].item()) - 1, start)
        token_mask[i, start:end] = 1.0

    return gathered, token_mask



def maybe_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
