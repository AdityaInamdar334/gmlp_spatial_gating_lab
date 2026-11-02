"""Small CPU benchmarks. These are NOT NeurIPS, ImageNet, or SQuAD replications."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import GMLPClassifier, AttentionClassifier, parameter_count
from .data import make_batch


def make_model(kind, length=12, width=48):
    if kind == "attention":
        return AttentionClassifier(length=length, width=width)
    if kind == "gmlp":
        return GMLPClassifier(length=length, width=width)
    if kind == "amlp":
        return GMLPClassifier(length=length, width=width, tiny_attention_dim=16)
    if kind == "no_mixing":
        return GMLPClassifier(length=length, width=width, disable_mixing=True)
    if kind == "random_init":
        return GMLPClassifier(length=length, width=width, init="random")
    raise ValueError(kind)


def fit_one(kind, task, seed=0, steps=220, batch_size=96, length=12,
            width=48, eval_size=768):
    torch.manual_seed(seed)
    model = make_model(kind, length, width)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.001)
    data_task = "content" if task == "content_ood" else task
    train_range = {"marker_start": 1, "marker_end": length//2+1} if task == "content_ood" else {}
    test_range = {"marker_start": length//2+1, "marker_end": length} if task == "content_ood" else {}
    start = time.perf_counter()
    for t in range(steps):
        x, y = make_batch(batch_size, length, data_task, seed=seed*1_000_000 + t + 11, **train_range)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_seconds = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        x, y = make_batch(eval_size, length, data_task, seed=100_000_000 + seed, **test_range)
        pred = model(x)
        accuracy = (pred.argmax(-1) == y).float().mean().item()
        cross_entropy = F.cross_entropy(pred, y).item()
        start = time.perf_counter()
        for _ in range(20):
            _ = model(x[:32])
        latency_ms = (time.perf_counter() - start)*1000/20
    return {"accuracy": round(accuracy, 5), "cross_entropy": round(cross_entropy, 5),
            "train_seconds": round(train_seconds, 3), "batch32_latency_ms": round(latency_ms, 3),
            "parameters": parameter_count(model)}


def benchmark(seeds=3, steps=220, batch_size=96, length=12, width=48,
              eval_size=768, output="results/benchmark.json"):
    torch.set_num_threads(min(2, torch.get_num_threads()))
    configurations = [
        ("fixed", "gmlp"), ("fixed", "attention"), ("fixed", "no_mixing"),
        ("content", "gmlp"), ("content", "attention"),
        ("content", "no_mixing"), ("content", "random_init"),
        ("content_ood", "gmlp"), ("content_ood", "amlp"), ("content_ood", "attention"),
        ("content_ood", "no_mixing")]
    records=[]
    for task, kind in configurations:
        for seed in range(seeds):
            metrics = fit_one(kind, task, seed, steps, batch_size, length, width, eval_size)
            record = {"task": task, "model": kind, "seed": seed, **metrics}
            records.append(record)
            print(f"{task:8} {kind:12} seed={seed}: accuracy={metrics['accuracy']:.3f} "
                  f"CE={metrics['cross_entropy']:.3f} train={metrics['train_seconds']:.1f}s", flush=True)
    summary=[]
    for task, kind in configurations:
        group = [r for r in records if r['task'] == task and r['model'] == kind]
        summary.append({"task":task,"model":kind,"accuracy_mean":round(statistics.mean(r['accuracy'] for r in group),5),
           "accuracy_std":round(statistics.stdev(r['accuracy'] for r in group),5) if len(group)>1 else 0,
           "ce_mean":round(statistics.mean(r['cross_entropy'] for r in group),5),
           "train_seconds_mean":round(statistics.mean(r['train_seconds'] for r in group),3),
           "batch32_latency_ms_mean":round(statistics.mean(r['batch32_latency_ms'] for r in group),3),
           "parameters":group[0]['parameters']})
    output_obj={"description":"Synthetic educational task; no pretrained model or benchmark reproduction.",
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},
        "settings":{"seeds":seeds,"steps":steps,"batch_size":batch_size,"length":length,
                    "width":width,"eval_size":eval_size},"summary":summary,"runs":records}
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output_obj,indent=2)+"\n")
    return output_obj


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=3)
    parser.add_argument("--steps",type=int,default=220)
    parser.add_argument("--batch-size",type=int,default=96)
    parser.add_argument("--length",type=int,default=12)
    parser.add_argument("--width",type=int,default=48)
    parser.add_argument("--eval-size",type=int,default=768)
    parser.add_argument("--output",default="results/benchmark.json")
    args=parser.parse_args()
    benchmark(args.seeds,args.steps,args.batch_size,args.length,args.width,args.eval_size,args.output)
