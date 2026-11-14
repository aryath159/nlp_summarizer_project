from fastapi import FastAPI
from pydantic import BaseModel
from backend.engine import SummarizerEngine

app = FastAPI(
    title="Dialogue Summarization & Action Extraction API",
    description="4-Bit QLoRA + DPO Fine-Tuned Qwen2.5-3B-Instruct API",
    version="1.0.0",
)

# Load model once at server startup
engine = SummarizerEngine()


class SummarizeRequest(BaseModel):
    dialogue: str
    max_new_tokens: int = 120
    use_hierarchical: bool = False
    top_k_extractive: int = 2


@app.get("/health")
def health_check():
    return {"status": "online", "model": "Qwen2.5-3B-Instruct (QLoRA + DPO)"}


@app.post("/summarize")
def summarize_endpoint(req: SummarizeRequest):
    if req.use_hierarchical:
        return engine.summarize_hierarchical(
            req.dialogue, max_new_tokens=req.max_new_tokens
        )
    return engine.generate_summary(
        req.dialogue, max_new_tokens=req.max_new_tokens, use_adapter=True
    )


@app.post("/compare-all")
def compare_all_endpoint(req: SummarizeRequest):
    extractive = engine.textrank_extractive(req.dialogue, top_k=req.top_k_extractive)
    base_res = engine.generate_summary(
        req.dialogue, max_new_tokens=req.max_new_tokens, use_adapter=False
    )
    if req.use_hierarchical:
        ft_res = engine.summarize_hierarchical(
            req.dialogue, max_new_tokens=req.max_new_tokens
        )
    else:
        ft_res = engine.generate_summary(
            req.dialogue, max_new_tokens=req.max_new_tokens, use_adapter=True
        )
    structured = engine.extract_structured_json(req.dialogue)

    return {
        "extractive_summary": extractive,
        "base_model": base_res,
        "finetuned_model": ft_res,
        "structured_json": structured,
    }