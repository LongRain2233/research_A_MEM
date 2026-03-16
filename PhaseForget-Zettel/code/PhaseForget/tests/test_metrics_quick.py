"""Quick sanity check for all 6 metrics."""
from phaseforget.evaluation.metrics import (
    EvalMetrics, compute_f1, compute_bleu1,
    compute_rouge_l, compute_rouge2, compute_meteor, compute_sbert,
)
from sentence_transformers import SentenceTransformer

pred = "Alice went to the support group on May 7"
ref  = "support group"

m = EvalMetrics()
m.f1_scores.append(compute_f1(pred, ref))
m.bleu_scores.append(compute_bleu1(pred, ref))
m.rouge_l_scores.append(compute_rouge_l(pred, ref))
m.rouge2_scores.append(compute_rouge2(pred, ref))
m.meteor_scores.append(compute_meteor(pred, ref))
sbert = SentenceTransformer("all-MiniLM-L6-v2")
m.sbert_scores.append(compute_sbert(pred, ref, sbert))

print(f"F1      = {m.avg_f1:.4f}")
print(f"BLEU-1  = {m.avg_bleu:.4f}")
print(f"ROUGE-L = {m.avg_rouge_l:.4f}")
print(f"ROUGE-2 = {m.avg_rouge2:.4f}")
print(f"METEOR  = {m.avg_meteor:.4f}")
print(f"SBERT   = {m.avg_sbert:.2f}")
print("All 6 metrics OK")
