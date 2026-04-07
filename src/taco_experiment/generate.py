"""Prompt building and code generation for Qwen2.5-Coder instruct models."""

import re
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from transformers import LogitsProcessorList

from .config import MODEL_NAME, TEMPERATURE, TOP_P, NUM_SAMPLES, MAX_NEW_TOKENS
from .p_less_processors import PLessLogitsProcessor, PLessNormLogitsProcessor

EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>", "<|im_end|>"]


def truncate_after_eof_strings(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)
    return text[:match.start()] if match else text


def build_prompt(sample):
    """Build a chat prompt from a TACO problem for an instruct model."""
    question = sample["question"]
    starter_code = sample.get("starter_code", "")

    try:
        input_output = json.loads(sample["input_output"])
        fn_name = input_output.get("fn_name")
    except (ValueError, TypeError):
        fn_name = None

    prompt = question
    if starter_code and len(starter_code.strip()) > 0:
        prompt += f"\n\nStarter code:\n```python\n{starter_code}\n```"

    if fn_name:
        prompt += f"\n\nUse Call-Based format. The function name should be: {fn_name}"
    elif not starter_code or len(starter_code.strip()) == 0:
        prompt += "\n\nUse Standard Input format. Read input from stdin and print output to stdout."

    prompt += "\n\nProvide only the Python code solution, no explanations."
    return prompt


def load_model(model_name=MODEL_NAME):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


DECODING_METHODS = ("top_p", "pless", "pless_norm")

_PLESS_PROCESSORS = {
    "pless": PLessLogitsProcessor,
    "pless_norm": PLessNormLogitsProcessor,
}


def generate_samples(model, tokenizer, prompt, n_samples=NUM_SAMPLES,
                     temperature=TEMPERATURE, top_p=TOP_P,
                     max_new_tokens=MAX_NEW_TOKENS,
                     decoding_method="top_p"):
    """Generate n_samples code solutions for a given prompt.

    Args:
        decoding_method: one of "top_p", "pless", "pless_norm".
            - "top_p": standard temperature + nucleus sampling.
            - "pless" / "pless_norm": hyperparameter-free p-less decoding
              (temp=1.0, no top_p; a LogitsProcessor applies the threshold).
    """
    messages = [
        {"role": "system", "content": "You are an expert competitive programmer. Write clean, correct Python code to solve the given problem."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[1]

    generate_kwargs = dict(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
    )

    if decoding_method == "top_p":
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    elif decoding_method in _PLESS_PROCESSORS:
        generate_kwargs["temperature"] = 1.0
        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [_PLESS_PROCESSORS[decoding_method]()]
        )
    else:
        raise ValueError(f"Unknown decoding_method: {decoding_method!r}. "
                         f"Must be one of {DECODING_METHODS}")

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generate_kwargs)

    generations = []
    for output in outputs:
        decoded = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        clean = truncate_after_eof_strings(decoded)
        clean = extract_code_block(clean)
        generations.append(clean)

    return generations


def extract_code_block(text):
    """Extract code from markdown code blocks if present."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def load_existing_generations(gen_path):
    """Load already-completed generations from JSONL checkpoint."""
    completed = {}
    gen_path = Path(gen_path)
    if gen_path.exists():
        with open(gen_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    completed[item["task_id"]] = item
    return completed


def generate_all(model, tokenizer, samples, n_samples=NUM_SAMPLES,
                 temperature=TEMPERATURE, top_p=TOP_P,
                 checkpoint_path=None, decoding_method="top_p"):
    """Generate solutions for all problems with incremental checkpointing.

    Saves each result as a JSONL line immediately after generation.
    On resume, skips already-completed task_ids.
    """
    completed = {}
    if checkpoint_path:
        completed = load_existing_generations(checkpoint_path)
        if completed:
            print(f"  Resuming: {len(completed)} problems already generated, {len(samples) - len(completed)} remaining")

    results = list(completed.values())

    for task_id, sample in tqdm(samples, desc="Generating"):
        if task_id in completed:
            continue

        prompt = build_prompt(sample)
        generations = generate_samples(
            model, tokenizer, prompt,
            n_samples=n_samples, temperature=temperature, top_p=top_p,
            decoding_method=decoding_method,
        )
        item = {
            "task_id": task_id,
            "prompt": prompt,
            "output": generations,
            "difficulty": sample["difficulty"],
        }
        results.append(item)

        if checkpoint_path:
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps(item) + "\n")

        print(f"  [gen] task_id={task_id} difficulty={sample['difficulty']} "
              f"samples={len(generations)} first_50_chars={generations[0][:50]!r}...")

    return results
