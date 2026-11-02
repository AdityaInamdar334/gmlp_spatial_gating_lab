# gMLP: Spatial Gating, Static Token Mixing, and Tiny Attention

A **small, reproducible PyTorch study** inspired by *Pay Attention to MLPs* (Liu et al., NeurIPS 2021) and the annotated labml.ai implementation.

This project is **not** an implementation of Google's full gMLP/BERT/ImageNet training pipeline and does not reproduce the paper's published results. It studies the difference between fixed spatial mixing and content-dependent attention on synthetic token-retrieval tasks.

## Questions

- Can a gMLP solve a token-selection task without any attention?
- What does a model learn when the relevant token is always at a known position?
- What happens when the relevant token moves to a position never observed during training?
- Can one small attention head improve transfer to those positions?
- What happens if we remove cross-token spatial mixing entirely?
- Does a triangular visibility mask actually prevent information leaking from the future?

## Quick start

Python 3.10+ is recommended. CPU is sufficient.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m gmlp_lab.demo
python -m gmlp_lab.benchmark --seeds 3 --steps 140 --batch-size 96 --eval-size 768 --output results/benchmark.json
python plot_results.py
```

For machines with many CPU threads, setting `OMP_NUM_THREADS=2` can make these tiny workloads more stable and reduce overhead.

No GPU, API key, external dataset, or Hugging Face checkpoint is required.

## Main benchmark

The inputs contain 12 tokens, including a query token at position zero. Each ordinary token has a binary value. The model must report the relevant bit from its query representation.

- **fixed:** Always retrieve the bit at position 3.
- **content:** Retrieve the bit at the uniquely marked position. Every marker position is present in training.
- **content_ood:** Train with the marker at positions 1 to 6. Test exclusively at positions 7 to 11.

We compare these implementations:

- `gmlp`: Two spatial-gated MLP blocks without attention.
- `attention`: Two-layer Transformer encoder, four attention heads and learned position embeddings.
- `amlp`: gMLP blocks augmented by a small single-head, 16-dimensional attention branch at the gate.
- `no_mixing`: Spatial matrices fixed to zero with bias one.
- `random_init`: gMLP spatial matrices with random Xavier initialization and zero gate bias.

These are **not parameter-matched architectures**. Their differences illustrate inductive biases in a small example, not which is optimal at equal compute.

## Results

Saved results are in `results/benchmark.json`, with per-seed records and aggregate means and sample standard deviations.

With three seeds and 140 optimizer steps:

- Fixed-position task: gMLP **100.0%**, attention **100.0%**, no spatial mixing **49.9%**.
- Content-based task with all positions observed: gMLP **100.0%**, attention **100.0%**, random-init gMLP **100.0%**, no spatial mixing **50.3%**.
- Previously unseen marker positions: gMLP **53.2%**, aMLP-style hybrid **100.0%**, attention **100.0%**, no spatial mixing **49.2%**.

Because the task is deliberately easy, perfect accuracy on an in-distribution split is not evidence of strong NLP generalization. The unseen-position experiment is the main diagnostic.

## Code layout

```text
gmlp_spatial_gating_lab/
  ARTICLE.md
  README.md
  BENCHMARK_NOTES.md
  requirements.txt
  LICENSE
  plot_results.py
  gmlp_lab/
    __init__.py
    models.py
    data.py
    benchmark.py
    demo.py
  tests/
    test_lab.py
  results/
    benchmark.json
    run.log
  figures/
    fixed_accuracy.png
    content_accuracy.png
    content_ood_accuracy.png
```

`models.py` has a batch-first Spatial Gating Unit, reversible causal mask support, small classifier stacks, an optional aMLP-style branch, and a Transformer baseline. `data.py` produces deterministic datasets. `benchmark.py` runs the baselines and ablations with independent seeds. `tests/test_lab.py` verifies shapes, initializations, gradients, mask behavior, future-token causality, dataset correctness, and training smoke tests.

## What the tests protect

The most important correctness test changes future input tokens and checks that earlier hidden states remain unchanged when a lower-triangular mask is applied. Another test differentiates early outputs with respect to later inputs and verifies the future gradients are zero. There is also a control proving that unrestricted mixing can allow future tokens to change earlier states.

These checks matter if you repurpose gMLP for autoregressive decoding. Simply using a next-token loss will not make an unmasked spatial mixer causal.

## Limitations

The benchmark is synthetic and tiny. It is **not** ImageNet, C4, SQuAD, GLUE, or a pretrained language model; it uses no real linguistic content and has no inference-time token generation. Models differ in parameter budgets. The content OOD split tests one positional extrapolation problem, not all forms of distribution shift. In particular, a larger or more structured gMLP may behave differently. Latency recordings are implementation and device specific.

See `BENCHMARK_NOTES.md` for complete settings and interpretation, and `ARTICLE.md` for the publication-ready tutorial.

## Primary sources

- Liu et al., [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050), NeurIPS 2021.
- [labml.ai annotated gMLP implementation](https://nn.labml.ai/transformers/gmlp/index.html).
- [PyTorch Image Models gMLP implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py).

## License

Independent educational code in this repository is licensed under MIT. Refer to the original sources for their own licenses and attribution requirements. The paper's results and source implementations are not repackaged as this project's code.
