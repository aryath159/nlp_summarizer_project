import json
import time
import torch
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./artifacts/summarizer-qlora-dpo-final"

SYSTEM_PROMPT = (
    "You are an expert dialogue summarizer. Provide a concise, factually accurate "
    "summary without swapping speaker roles."
)


class SummarizerEngine:
    
    def __init__(self, adapter_path: str = ADAPTER_DIR):
        print("Loading tokenizer and base model (CPU / MPS mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Automatically select Mac GPU (mps) or Laptop CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Use bfloat16 on CPU to cut RAM usage from 12GB down to ~6GB!
        dtype = torch.float16 if device == "mps" else torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        print(f"Attaching LoRA + DPO adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print(f"Model loaded successfully on {device.upper()}!")

    def textrank_extractive(self, dialogue: str, top_k: int = 2) -> str:
        """Classical unsupervised graph baseline using TF-IDF + PageRank."""
        lines = [line.strip() for line in dialogue.split("\n") if line.strip()]
        if len(lines) <= top_k:
            return " ".join(lines)
        tfidf = TfidfVectorizer().fit_transform(lines)
        sim_matrix = cosine_similarity(tfidf)
        np.fill_diagonal(sim_matrix, 0)
        scores = nx.pagerank(nx.from_numpy_array(sim_matrix))
        top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:top_k])
        return " ".join([lines[i] for i in top_indices])

    def generate_summary(
        self,
        dialogue: str,
        max_new_tokens: int = 120,
        use_adapter: bool = True,
    ) -> dict:
        """Generates a summary with either the fine-tuned adapter or zero-shot base model."""
        start_time = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize the following conversation:\n\n{dialogue}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_token_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if use_adapter:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                with self.model.disable_adapter():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

        generated_ids = output_ids[0][input_token_len:]
        summary_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency = round(time.perf_counter() - start_time, 2)

        return {
            "summary": summary_text,
            "input_tokens": input_token_len,
            "output_tokens": len(generated_ids),
            "compression_ratio": round(len(generated_ids) / max(input_token_len, 1), 2),
            "latency_sec": latency,
        }

    def summarize_hierarchical(
        self,
        long_dialogue: str,
        chunk_token_limit: int = 600,
        overlap_lines: int = 2,
        max_new_tokens: int = 120,
    ) -> dict:
        """Map-Reduce hierarchical pipeline for long transcripts."""
        start_time = time.perf_counter()
        lines = [l for l in long_dialogue.split("\n") if l.strip()]
        chunks, current_chunk, current_len = [], [], 0

        for line in lines:
            line_len = len(self.tokenizer.encode(line, add_special_tokens=False))
            if current_len + line_len > chunk_token_limit and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = current_chunk[-overlap_lines:]
                current_len = sum(
                    len(self.tokenizer.encode(l, add_special_tokens=False))
                    for l in current_chunk
                )
            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        if len(chunks) == 1:
            res = self.generate_summary(chunks[0], max_new_tokens=max_new_tokens, use_adapter=True)
            res["num_chunks"] = 1
            res["chunk_summaries"] = [res["summary"]]
            return res

        # MAP Step
        chunk_summaries = [
            self.generate_summary(c, max_new_tokens=80, use_adapter=True)["summary"]
            for c in chunks
        ]
        # REDUCE Step
        combined_text = "\n".join(
            [f"#Person1#: Part {i+1}: {s}" for i, s in enumerate(chunk_summaries)]
        )
        final_res = self.generate_summary(
            combined_text, max_new_tokens=max_new_tokens, use_adapter=True
        )

        return {
            "summary": final_res["summary"],
            "num_chunks": len(chunks),
            "chunk_summaries": chunk_summaries,
            "latency_sec": round(time.perf_counter() - start_time, 2),
        }

    def extract_structured_json(self, dialogue: str) -> dict:
        """Extracts structured JSON with summary, participants, action items, and topic."""
        json_system_prompt = (
            "You are an AI meeting assistant. Analyze the dialogue and output ONLY a valid JSON object "
            "with these exact keys: "
            '{"summary": "concise overview", "participants": ["list of speakers"], '
            '"action_items": ["specific tasks and owners"], "topic": "core subject"}.'
        )
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": dialogue},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            with self.model.disable_adapter():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        raw = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw_output": raw}