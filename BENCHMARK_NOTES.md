# Reproducibility and Benchmark Interpretation

This educational benchmark is not affiliated with Google Research, NeurIPS, or labml.ai.

## Experimental design

- **Environment:** Python 3.11, PyTorch 2.10.0+cpu for the saved reference run. See `results/benchmark.json` for the actual versions captured.
- **Device:** CPU. PyTorch thread count bounded to at most two in the CLI benchmark.
- **Seeds:** 0, 1, 2.
- **Training steps:** 140 with AdamW, learning rate 0.002, weight decay 0.001, gradient clip norm 1.
- **Batch size:** 96.
- **Held-out evaluation size:** 768 per seed and configuration.
- **Input:** 12 positions, five real-valued channels, one query token and eleven value tokens.
- **Target:** Binary class equal to the bit at a known position or at the unique marked position.
- **gMLP:** width 48, expansion 96, two blocks, fixed learned `12 x 12` mixing matrix per block.
- **aMLP-style hybrid:** same gMLP with a single 16-dimensional content-dependent attention head contributing to the gate in each block.
- **Attention baseline:** width 48, two Transformer encoder layers, four attention heads, 96-wide feedforward expansion, learned positional embeddings.
- **Metrics:** Held-out accuracy, held-out cross-entropy, parameter counts, measured CPU training and forward-loop time.

## Dataset separation

All train batches are generated with deterministic seeds in the training seed range. The evaluation seed range begins at 100,000,000, so the test inputs do not reuse training draws.

For `content_ood`, train marker positions are 1 through 6 and held-out marker positions are 7 through 11. The target function, binary value at the marked position, is unchanged. No test-marker examples occur in the training generator in that configuration.

## Saved results, three seeds

**Fixed position:**

- gMLP: 100.00% average accuracy.
- Attention: 100.00%.
- No mixing: 49.91%.

**Content marker, positions seen during training:**

- gMLP: 100.00%.
- Attention: 100.00%.
- No mixing: 50.30%.
- Random-init gMLP: 100.00%.

**Held-out marker positions:**

- gMLP: 53.21% mean, 4.73 percentage points sample standard deviation.
- aMLP-style hybrid: 100.00%.
- Attention: 100.00%.
- No mixing: 49.18%.

The results do not demonstrate that a model is generally superior. They reveal one structured distribution shift and two inductive-bias choices under a specific capacity and optimization budget.

## Model parameter counts

- gMLP: 15,290 total parameters.
- aMLP-style hybrid: 21,626 total parameters.
- Transformer baseline: 38,978 total parameters.

The no-mixing model still reports frozen spatial parameters in its total count. A more rigorous efficiency study should report both total and trainable parameters and match architectures by quality, compute, and parameter count.

## Timing interpretation

CPU timings are not reliable measures of GPU efficiency or real-world model speed. They reflect small eager-mode PyTorch networks, batch size 32 for evaluation timing, machine-specific implementation details, and background load. Do not use them to make claims about published gMLP versus Transformer efficiency.

## Tests and causal guarantees

The test suite checks shapes, initializations, gradients, frozen mixing parameters, dataset labels, deterministic data, masked position mixing, and causal leakage. An intervention test changes future token values and confirms earlier representations are invariant under a triangular spatial mask. A gradient test checks the absence of future input derivatives for a selected earlier output.

These are implementation tests, not a formal verification of a generative language model.

## Suggested follow-up research

Compare parameter-matched or training-FLOP-matched models, add a structured Toeplitz spatial matrix, include a learned relative-position attention baseline, extend to variable-length sequences, and evaluate multiple transfer axes such as shifting a relevant position, extending sequence length, and changing the marker encoding. Avoid interpreting a synthetic binary retrieval task as a general statement about language understanding.
