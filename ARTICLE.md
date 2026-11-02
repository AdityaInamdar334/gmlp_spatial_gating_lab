# Google Removed Self-Attention From a Transformer. The Model Still Learned. Then the Tokens Moved.

*The gMLP experiment that challenged Transformer orthodoxy, explained from the original paper and labml.ai PyTorch code, with a reproducible study of spatial gating, causal masks, content-based retrieval, and when tiny attention earns its keep.*

**Reading time:** A full technical tutorial, intended for machine learning engineers and researchers.

In 2021, a team at Google Research asked an uncomfortable question: what if the component everybody credited for the Transformer's success was not always necessary?

Self-attention had become the default language for explaining modern neural networks. Tokens could search for other tokens. The resulting attention matrices changed with every input. That adaptive routing seemed central to everything from image recognition to masked-language modeling.

Then Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le introduced **gMLP**, a stack of gated multilayer perceptrons with **no self-attention at all**. Their spatial projection matrices were learned during training but remained fixed for every example at inference. The resulting network reached **81.6% top-1 ImageNet accuracy** in its gMLP-B configuration, competitive with the reported **81.8%** for DeiT-B in the paper's comparison. On masked-language-model pretraining, larger gMLPs approached or exceeded similarly sized Transformer baselines in perplexity. [1]

Yet that was not the end of the story.

For SQuAD v2.0, the paper's gMLP-base reached **70.1 F1**, while its matched experimental BERT-base reached **78.6**. Adding just **one 64-dimensional attention head** to the gating mechanism produced a hybrid named aMLP, which reached **80.9 F1**. The pretraining scores had been similar. The downstream ability to align information was not. [1]

That tension is what makes gMLP worth studying today. It tells us something deeper than whether an MLP can beat a Transformer. It separates **having a way for tokens to communicate** from **deciding, based on the current input, which tokens ought to communicate**.

We will build both mechanisms from scratch, check the spatial gating code against the labml.ai implementation, and run a small experiment in which the answer moves from a familiar token position to a previously unseen one. The model that learned token retrieval without attention will suddenly struggle. A tiny content-aware attention branch will recover.

The experiment does not replicate Google's results. It isolates one plausible reason their hybrid architecture helped.

---

## Table of contents

1. The question behind *Pay Attention to MLPs*
2. The three operations inside a gMLP block
3. Why a regular MLP cannot exchange token information
4. Spatial Gating Unit, from tensor dimensions to code
5. Why the gate starts close to one
6. How the labml.ai implementation works
7. The difference between static mixing and dynamic attention
8. Build the complete gMLP block in PyTorch
9. The causal-mask trap in autoregressive gMLP
10. Why position matters, even without positional embeddings
11. What the original paper actually measured
12. Why pretraining perplexity can mislead you
13. How a tiny attention head becomes aMLP
14. Design a controlled token-retrieval benchmark
15. Build the synthetic dataset
16. Train gMLP, attention, and ablation baselines
17. Results on familiar positions
18. The distribution-shift experiment
19. The tiny-attention recovery
20. Ablation: what happens without spatial mixing
21. Ablation: near-zero versus random initialization
22. Test the causal implementation instead of trusting it
23. Latency, parameter count, and hardware caveats
24. How to reproduce the experiment
25. Limitations, failure modes, and a better research agenda
26. What gMLP still teaches us
27. References

---

## 1. The question behind *Pay Attention to MLPs*

The original paper, *Pay Attention to MLPs*, was published at NeurIPS 2021 by researchers from Google's Brain Team. [1]

Its starting observation was surprisingly modest. A Transformer contains two broad kinds of computation. Its attention sublayer exchanges information among tokens, and its feed-forward sublayer transforms each token's feature vector. The latter is already an MLP. Could one replace the attention sublayer with a sufficiently well-designed spatial mixing function, retaining the advantages of parallel token processing and deep residual stacks?

This question mattered because architectures tend to accumulate assumptions. Once a design becomes dominant, researchers often attribute too much of its success to its most distinctive component. The paper instead treated attention as an **inductive bias to test**, not an indispensable ingredient.

The authors' answer was nuanced.

For ImageNet classification, a gating-based MLP could match strong attention-based vision baselines under the reported setup. For BERT-style masked-language pretraining, it could achieve very similar perplexity. But some downstream NLP tasks, especially those involving interactions between separate spans of text, benefited from adaptive token-to-token alignment.

That distinction remains important for model design. The right question is not merely whether an architecture contains attention. It is whether its token-mixing operation supplies the appropriate inductive bias for the data distribution and the task.

## 2. The three operations inside a gMLP block

Take a sequence with `N` tokens and a hidden width of `D` features per token.

A gMLP block performs three main steps:

1. **Channel expansion.** Apply a learned linear projection independently to each token, followed by GeLU. The hidden width becomes an expanded width `E`.
2. **Spatial gating.** Split the expanded representation into two equal halves. Use a shared `N x N` matrix to mix token positions in one half, then use the resulting values to gate the other half.
3. **Channel contraction and residual addition.** Project the gated width `E / 2` back to `D`, then add the original input.

The complete block can be summarized without any special notation:

```text
Input: [batch, tokens, D]
        |
        v
     LayerNorm
        |
        v
  Linear(D -> E)
        |
       GeLU
        |
        v
  Split into A and B
     /       \
    A         B
    |      LayerNorm
    |         |
    |   Spatial N x N mixing
    |         |
    |      Add bias 1
    |         |
     \       /
      A * gate(B)
           |
    Linear(E/2 -> D)
           |
     Add input residual
           |
Output: [batch, tokens, D]
```

There is no query-key similarity matrix in this path. No token computes attention weights from the current example. Nevertheless, every output position can depend on information from every input position, because the spatial matrix connects token locations.

## 3. Why an ordinary MLP cannot do the same job

Consider the sequence:

```text
[QUERY] [blue] [green] [red] [yellow]
```

Suppose the model must answer a question at `QUERY` using the token at position three. A regular feed-forward network applied independently to each token cannot move the information stored at position three into the representation at position zero.

Adding more tokenwise feed-forward layers does not solve this. If each layer acts separately on each position, the query representation never receives the other tokens. The architecture needs a cross-token path.

A Transformer constructs that path dynamically. It computes query and key vectors from the input, scores relationships among tokens, and aggregates value vectors.

A gMLP constructs the path through **learned spatial connections**. Row zero of its spatial matrix controls which positions affect token zero. The same row is used regardless of the words or numbers currently occupying those positions.

But gMLP is not merely a linear spatial projection. The spatially mixed result multiplies another feature stream. That multiplicative interaction gives the architecture more expressive cross-token computations than a static linear mixer alone.

## 4. The Spatial Gating Unit, one shape at a time

The code in the labml.ai tutorial uses tensors ordered as `[sequence, batch, channels]`. Our companion lab uses the more familiar `[batch, sequence, channels]`, but the computation is the same. [2]

Start with an activated expansion of shape `[B, N, E]`. `E` must be even because the last dimension is split:

```python
carry, gate_input = expanded.chunk(2, dim=-1)
```

Both tensors have shape `[B, N, E/2]`.

Normalize `gate_input` along its feature dimension. Then apply a learnable matrix with shape `[N, N]` across sequence positions. Crucially, **the same spatial matrix is applied independently to every channel** in the gate stream.

For every output token position `i`, the gate collects a weighted mixture of all input token positions `j`. The resulting gate is shifted by a learned bias. The output multiplies this gate with the carry stream element by element.

In PyTorch, the essential operation is:

```python
mixed = torch.einsum("ij,bjc->bic", weight, gate_input)
gated = carry * (mixed + bias[None, :, None])
```

Read that `einsum` carefully. Its first operand has token indices `i` and `j`. Its second operand has batch `b`, input position `j`, and channel `c`. The result preserves `b` and `c`, while combining the `j` positions into each output `i`.

This is spatial mixing, not channel mixing. The channel projection happened earlier, and the gate's token mixing matrix is shared among the gate channels.

### Why multiply instead of add?

Suppose a position contributes a positive amount of evidence to some feature. An additive mixer always adds its contribution. A multiplicative gate can instead scale the current feature, suppress it, amplify it, or reverse its sign depending on the gate value.

The original paper compared linear, additive, multiplicative, and split multiplicative alternatives. In its masked-language modeling ablation, the **split multiplicative SGU** had a validation perplexity of **4.35**, outperforming the tested additive version at **4.97** and the plain linear version at **5.14**. [1]

That is an empirical result under the authors' training setup, not a universal theorem that multiplication always wins.

## 5. Why the learned gate starts near one

One of the best engineering decisions in gMLP is also easy to overlook.

The authors initialize the spatial mixing matrix with values near zero and its bias near one. The initial gate therefore starts near `1`.

At the beginning of training, multiplying the carry stream by the gate leaves it nearly unchanged. The network initially behaves more like a stable tokenwise feed-forward block. As training progresses, the spatial matrix learns to exchange information among positions.

A random spatial matrix with a zero bias behaves differently. It may impose strong and poorly calibrated cross-token interactions immediately. In deep networks, this can complicate optimization.

The original paper calls this initialization important for training stability. Our small lab includes a randomized-initialization ablation, but it should not be used to claim that one initialization is better in every regime. In our short, simple task, both settings eventually solve the task.

This is a useful example of why a model's initial function can matter as much as its parameter count.

## 6. Reading the labml.ai implementation without getting lost

The implementation you supplied follows the paper's design closely. [2]

Its `GMLPBlock` contains a pre-normalization layer, `proj1`, GeLU, a `SpacialGatingUnit`, `proj2`, and the residual connection.

`proj1` increases the channel dimension to `d_ffn`. The gating unit halves it through the split. `proj2` therefore maps from `d_ffn // 2` back to `d_model`.

Inside `SpacialGatingUnit`, the learned `weight` has shape `[seq_len, seq_len]`, and `bias` has shape `[seq_len]`. The implementation slices both down to the length of the current input, which permits a shorter input than the maximum configured length.

The line to understand is:

```python
torch.einsum('ij,jbd->ibd', weight, z2)
```

This is exactly the same spatial projection as our batch-first implementation, just with `j` appearing first in `z2` because the tensor layout is `[tokens, batch, channels]`.

The implementation also accepts a visibility mask. Its third mask dimension is constrained to `1`, allowing the same positional mask to be shared across every sample in a batch.

One small naming detail: the reference class is spelled `SpacialGatingUnit`, but the standard term in the paper is **Spatial Gating Unit**.

The tutorial on labml.ai is a valuable annotated reference, but the code does not by itself reproduce the paper's billion-token-scale pretraining, ImageNet results, or SQuAD experiments. It teaches the building block.

## 7. Static token mixing is not the same as static behavior

This distinction matters enough to repeat.

**The gMLP spatial matrix is independent of the input, but the gMLP output is not.**

The spatial matrix supplies fixed positional connections. The normalized gate input changes with each example. The carry stream also changes. Multiplication allows different inputs to produce very different gated outputs using the same position matrix.

So it would be incorrect to call gMLP a model that cannot do content-sensitive computation. Our own benchmark will show that gMLP can learn to select a marked token based on input content.

What it lacks is a straightforward equivalent to attention's **input-dependent pairwise routing weights**. In attention, the connection strength between position `i` and position `j` depends on their current representations. In the SGU, the learned spatial coefficient between those two positions does not.

As a result, a gMLP may need to learn separate positional communication patterns for cases an attention mechanism handles with a shared content-based matching rule.

This is an inductive-bias difference, not an expressivity impossibility theorem.

## 8. Build the complete block in PyTorch

Here is the minimal core of our implementation. The version in the ZIP adds length and shape validation, optional masking, an initialization ablation, and a hybrid attention branch.

```python
import torch
from torch import nn

class SpatialGatingUnit(nn.Module):
    def __init__(self, expanded_dim, seq_len):
        super().__init__()
        assert expanded_dim % 2 == 0
        self.norm = nn.LayerNorm(expanded_dim // 2)
        self.weight = nn.Parameter(
            torch.empty(seq_len, seq_len).uniform_(-0.01, 0.01)
        )
        self.bias = nn.Parameter(torch.ones(seq_len))

    def forward(self, z, mask=None):
        n = z.shape[1]
        carry, gate = z.chunk(2, dim=-1)
        gate = self.norm(gate)
        w = self.weight[:n, :n]
        if mask is not None:
            w = w * mask
        gate = torch.einsum('ij,bjc->bic', w, gate)
        gate = gate + self.bias[:n][None, :, None]
        return carry * gate

class GMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.proj2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x, mask=None):
        z = self.act(self.proj1(self.norm(x)))
        return x + self.proj2(self.sgu(z, mask))
```

Try it with a batch of four sequences, each containing twelve tokens with 32 features:

```python
block = GMLPBlock(d_model=32, d_ffn=64, seq_len=12)
x = torch.randn(4, 12, 32)
y = block(x)
print(y.shape)
# torch.Size([4, 12, 32])
```

The block preserves input shape because its learned transformation is added to the residual.

There are two architectural constraints worth recording. First, `d_ffn` must be even. Second, the spatial matrix is tied to a configured maximum sequence length. A larger `N` requires additional parameters or a different parameterization, such as a structured kernel.

## 9. The causal-mask trap

The labml.ai page also presents an autoregressive model built with gMLP blocks. For autoregressive generation, position `t` must never depend on tokens after `t`.

An unconstrained spatial matrix violates that requirement. If `W[t, t+1]` is nonzero, information from the next token can flow into the representation at `t`. Training such a model with next-token labels can leak answers from the future.

The simplest defense is a **lower-triangular mask**:

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
output = block(x, mask=mask)
```

Every row `i` can then combine only positions `j` where `j` is at most `i`. The mask applies to the spatial weights, not just to the model's loss function.

For a stack of correctly masked gMLP blocks, position `i` remains independent of future inputs because each block contains only tokenwise channel operations, masked spatial mixing, and a residual connection.

But you should not trust the shape of the mask alone. You should test it causally. We will do that later by editing future inputs and verifying that earlier outputs remain unchanged.

This is particularly important when adapting image or masked-language code to autoregressive language modeling. Image classification and bidirectional masked-language modeling are allowed to combine information from both directions. Autoregressive decoding is not.

## 10. gMLP encodes positions in its weights

The paper's gMLP does not need explicit position embeddings for the tasks it studies. The spatial matrix already distinguishes input position zero from position seven, because those positions correspond to different rows and columns of `W`. [1]

That is convenient, but it creates an unusual tradeoff.

Attention without position information is largely permutation-equivariant across the tokens. When a model needs order, we add positional signals such as absolute embeddings, relative biases, or rotary encodings.

gMLP arrives with absolute positional structure built into the token-mixing matrix. It is naturally position aware, but its learned positional connections may become strongly associated with where a feature appeared in training.

The original authors also found that, for masked-language modeling, learned spatial matrices tended toward **Toeplitz-like patterns**, meaning similar mixing weights could recur at equal relative offsets. They constrained this structure in some experiments to remove redundant parameters. [1]

That is an important bridge between gMLP and convolution. Convolutions share filters across positions and are therefore built for shift-related patterns. An unconstrained `N x N` spatial matrix does not automatically provide the same guarantee.

## 11. What the NeurIPS paper actually measured

It is easy to cherry-pick the most impressive number from *Pay Attention to MLPs* and conclude that attention was proven unnecessary. The paper's full picture is more interesting.

### ImageNet

The paper reports the following results in its 224-pixel comparison:

- gMLP-Ti: **72.3%** top-1 accuracy with about **6 million** parameters.
- gMLP-S: **79.6%** with about **20 million** parameters.
- gMLP-B: **81.6%** with about **73 million** parameters.
- DeiT-B: **81.8%** with about **86 million** parameters.

These comparisons suggest that learned spatial gating can be a serious alternative to vision attention under the reported training conditions. They do not imply that every gMLP is faster or better than every Vision Transformer. [1]

### Masked-language pretraining

In the paper's smaller-scale comparison, the relative-position Transformer had validation perplexity **4.26** and a split-gated gMLP had **4.35**. A strong result for a model without attention.

When both architectures were enlarged, the gMLP perplexity became competitive with or slightly better than the comparable Transformer in the paper's reported configurations. But the relevant models did not always have identical compute requirements or training behavior.

### Language understanding and question answering

For the full BERT-style setup, where the authors used large-scale masked-language pretraining on English C4, the reported dev-set F1 scores on SQuAD v2.0 were:

- BERT-base trained in the authors' setup: **78.6**.
- gMLP-base: **70.1**.
- aMLP-base with a single 64-dimensional attention head: **80.9**.
- BERT-large trained in the authors' setup: **81.0**.
- gMLP-large: **78.3**.
- aMLP-large with a 128-dimensional attention head: **85.4**.
- gMLP-xlarge without attention: **82.1**, using substantially more parameters.

The base gMLP falls **8.5 F1 points** below the paper's BERT-base experimental baseline, while the tiny-attention hybrid ends **2.3 points above** it. The larger hybrid similarly outperforms the authors' BERT-large baseline, although the architecture and FLOP budgets are not identical. [1]

These figures should be attributed to the paper. They are not results from the code in this tutorial.

## 12. Similar perplexity does not mean similar reasoning

Perplexity measures how well a model predicts withheld language tokens. It is a valuable pretraining metric, but a model can achieve similar average perplexity to another while organizing information differently.

Consider two sentences containing names, entities, and conflicting relationships. A question-answering model must connect a question phrase with the specific supporting span in the other sentence, possibly while rejecting tempting nearby distractors.

Attention supplies an explicit content-dependent matching operation that can associate one token with another. gMLP has to produce useful alignment through static positional mixing plus deep nonlinear transformations. That may be possible, especially at scale, but it may require more capacity or data.

The paper's results illustrate this separation. gMLP performs strongly on the single-sentence SST-2 classification task, while lagging behind the corresponding Transformer on some sentence-pair and question-answering tasks. The authors suggest that cross-sentence alignment is part of the explanation. [1]

This matters whenever an architecture is justified using only a proxy metric. A language model's ability to predict tokens should not be treated as a complete measurement of retrieval, compositional reasoning, document grounding, or robustness to altered token positions.

## 13. One attention head changes the calculation

The authors' hybrid **aMLP** introduces a small content-dependent attention branch alongside the spatial gate.

It does not replace every gMLP block with a Transformer. Instead, it computes a single-head attention result from the normalized block input and feeds the result into the gating pathway. The paper tests attention widths as small as 64, with 128 in larger versions. [1]

Why might this help?

Suppose a question token is searching for the sentence fragment containing a particular entity. With a static spatial matrix, the model must learn useful paths through positions and feature interactions. With attention, the query token can score potential matching keys **for the current example**.

The model can therefore retain gMLP's broad spatial mixing while adding an inexpensive route for content-based alignment.

Our companion code implements an **aMLP-style** branch using one attention head of width 16, which is deliberately smaller than the paper's models:

```python
class TinyAttention(nn.Module):
    def __init__(self, width, output_width, head_dim=16):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * head_dim)
        self.out = nn.Linear(head_dim, output_width)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        weights = scores.softmax(-1)
        return self.out(weights @ v)
```

In our full code, the output of this branch is added to the SGU gate before multiplying the carry features. This is a pedagogical hybrid inspired by the paper, not a claim of byte-for-byte compatibility with Google's original training code.

## 14. A benchmark that exposes the inductive bias

We want a compact experiment that both gMLP and attention can learn, but that exposes their differences when the data distribution changes.

The problem is binary token retrieval.

Each example has twelve positions:

```text
[QUERY] [token 1] [token 2] ... [token 11]
```

Each ordinary token contains a binary value. Some examples also mark exactly one token as relevant. The model must predict the value at the requested position and place its answer in the `QUERY` representation.

We construct three tasks.

**Task A: Fixed position.** The answer is always the bit at token position three. Position three is visible during training and testing.

**Task B: Content selection.** Each example marks one of eleven token positions. The answer is the bit at that marked position. All possible marker positions appear in training.

**Task C: Unseen-position shift.** The model trains only when the marker appears at positions one through six, then is evaluated only when the marker appears at positions seven through eleven.

For Task C, the rule never changes: return the bit belonging to the marked token. Only the positions at which the marker can appear change.

We compare three architectures and two ablations:

- gMLP with two spatial gating blocks and fixed learned spatial matrices.
- A two-layer Transformer encoder with content-dependent self-attention and learned position embeddings.
- The aMLP-style gMLP with a single 16-dimensional attention head in each block.
- A gMLP with all spatial mixing weights frozen to zero, to demonstrate that channel-only processing cannot solve the query.
- A gMLP with a conventional random spatial initialization, to investigate the effect of the near-identity gate setup.

The models are **not parameter matched**. That is a deliberate limitation of this educational experiment. The aim is to isolate a failure mode and test a hybrid intervention, not to establish a universal efficiency ranking.

## 15. Generate the data without a dataset download

Every example is created locally from a deterministic seed. No language-model API, benchmark server, or third-party dataset is required.

The relevant features are:

- Two channels encode the binary token value.
- One channel marks the selected token for content-based retrieval.
- Two small nuisance channels make the examples less degenerate.
- The query token is distinct and does not itself reveal the answer.

The target label is calculated directly from the generated token sequence. This avoids using a model to label our own evaluation data.

Here is the concept in a simplified form:

```python
import torch

batch_size = 4
seq_len = 12
x = torch.zeros(batch_size, seq_len, 5)

# Every non-query token gets a random binary value.
bits = torch.randint(0, 2, (batch_size, seq_len - 1))
x[:, 1:, 0] = (bits == 0).float()
x[:, 1:, 1] = (bits == 1).float()

# Choose one marked token for each example.
marked = torch.randint(1, seq_len, (batch_size,))
x[torch.arange(batch_size), marked, 2] = 1.0

# Supervision comes from the marked token's bit.
y = bits[torch.arange(batch_size), marked - 1]
```

The full implementation also adds a fixed-position task, nuisance features, deterministic random generators, and explicit marker-position ranges. It is available in `gmlp_lab/data.py` in the companion project.

The critical detail for Task C is that training and test marker ranges are disjoint. We do not accidentally train on the positions we later describe as unseen.

## 16. Train the three model families

We use the same task generator, 140 optimizer steps, batch size 96, and the same held-out evaluation size of 768 for each configuration. We repeat the experiment with seeds zero, one, and two.

The compact models use a hidden width of 48 and two layers. The gMLP expands to width 96 before the spatial gate. The attention baseline uses four heads. The hybrid uses a single 16-dimensional head inside each gMLP block.

Each model minimizes cross-entropy between the predicted binary label and the known correct label.

The training loop is deliberately unremarkable:

```python
for step in range(140):
    x, labels = make_batch(
        size=96,
        length=12,
        task='content',
        seed=step + 11,
    )

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

We keep the architecture, optimization, dataset generation, and evaluation code open so that readers can change one assumption at a time.

## 17. What happened on familiar positions

The first two experiments are intentionally straightforward.

On fixed-position retrieval, **gMLP reached 100.0% mean held-out accuracy**, and the attention baseline also reached **100.0%**. The no-spatial-mixing ablation remained near random guessing at **49.9%**.

On content-based retrieval, where the marker could appear anywhere during training, **gMLP again reached 100.0%**, and the attention baseline reached **100.0%**. The no-mixing ablation reached **50.3%**.

The randomized-initialization gMLP also reached **100.0%** after the full 140-step schedule on this easy task.

These results establish two points.

First, a static spatial mixer plus nonlinear gating can indeed learn content-dependent behavior. The marked token changes from example to example, and gMLP still returns the correct bit.

Second, the tokenwise ablation cannot solve a problem requiring information from another position. The spatial operation is doing real work.

The result does **not** tell us that gMLP and attention have learned equally transferable representations. Both architectures have seen every marker position that will occur during evaluation.

## 18. Then we moved the marker

This is the most instructive experiment in the project.

Train the models on content retrieval, but restrict the marker to the first six ordinary-token positions. When training ends, generate a held-out test set in which the marker is restricted to the final five positions.

The target computation has not changed. A good abstract algorithm remains: find the marker, extract the associated value, return that value.

The results change sharply.

Over three seeds, the plain gMLP achieves **53.2% mean accuracy**, with **4.7 percentage points** of sample standard deviation. That is barely above chance on the new positions.

The full attention baseline reaches **100.0%** in the same experiment.

Why? One plausible explanation is that the learned gMLP spatial matrix has strengthened pathways from familiar marker positions toward the query. Previously unseen columns of its spatial projection have not received the same task-specific learning signal. Its content-sensitive gate does not automatically make the underlying positional routing rule transferable.

The attention baseline computes similarity between token representations. Its rule for selecting the marked token can transfer to a new position because the scoring mechanism is shared across positions. The learned positional embeddings in this particular model did not prevent that transfer.

It is important to be precise here: this is an **observed behavior in our small controlled benchmark**. It does not prove that every gMLP fails under positional shift or that attention is always superior. Better regularization, different training distributions, a structured spatial matrix, more data, or a larger gMLP might change the result.

The result nevertheless makes the inductive-bias difference visible.

## 19. A tiny attention branch restores transfer

The hybrid aMLP-style model uses the same two gMLP blocks but adds a single 16-dimensional attention head to each gate.

In the unseen-position test it achieves **100.0% mean held-out accuracy across three seeds**.

The hybrid uses **21,626 parameters** in our implementation, compared with **15,290** for the plain gMLP and **38,978** for the attention baseline.

This is not an argument that the hybrid is optimal, or even that it would remain superior on bigger problems. It is a mechanistic demonstration of the paper's central idea: **a small amount of content-dependent routing can complement static spatial mixing**.

Notice that the hybrid does not need to discard the gMLP structure. It retains channel expansion, GeLU, spatial gating, and residual connections. The attention branch supplies additional information about which current positions are relevant.

In a sense, the gMLP experiment in 2021 was not simply an argument against attention. It also helped identify one of attention's most valuable roles.

## 20. Remove the spatial matrix entirely

Our `no_mixing` ablation freezes every spatial matrix to zero and keeps the gate bias at one.

In this configuration, the gating block can still perform nonlinear transformations of each token's feature vector. It has a residual connection, layer normalization, channel expansion, and contraction.

What it no longer has is **any communication between token positions**.

Because the classifier reads only the query token at position zero, and the target bit lives in a different token, this ablation should have no access to the answer beyond random statistical correlations. The observed results, around 50%, are consistent with that reasoning.

This is a useful sanity check. If the no-mixing baseline somehow achieved perfect accuracy, we would investigate whether the label leaked into the query representation, whether the dataset was imbalanced, or whether an unintended path carried information across positions.

A proper baseline is not just a competitor. It is also an instrument for catching flaws in the benchmark.

## 21. What the initialization ablation tells us

The original gMLP paper emphasizes initializing the spatial matrix near zero and its bias at one, to help the network begin as an approximately tokenwise feed-forward model. [1]

Our companion lab offers a `random_init` mode that replaces the small spatial initialization with a conventional Xavier-style matrix and a zero bias.

On the simple content-retrieval task, both settings eventually achieve 100% accuracy with 140 steps and the chosen learning rate.

That does **not** refute the original paper's stability finding. Our experiment is tiny, shallow, and short. It has no hundreds-of-millions-of-parameters model, no language data, no deep residual optimization, and no need to preserve useful pretrained representations.

A meaningful stability investigation would compare training-loss trajectories, gradient norms, sensitivity to learning rate, failure rates across seeds, and depth scaling. Our experiment only establishes that the stable initialization is not necessary to solve this particular easy task under these conditions.

This is why ablations must be interpreted in the regime in which they were measured.

## 22. Test causality directly

A causal language model should not change earlier hidden states when only future input tokens change.

The lab includes two complementary tests.

**Intervention test.** Generate an input sequence. Copy it. Change only positions after a chosen cutoff. Run the masked gMLP on both inputs. The outputs before the cutoff must match.

**Gradient test.** Differentiate an early output with respect to the entire input. All derivatives with respect to future positions must be zero.

The minimal intervention test looks like this:

```python
import torch
from gmlp_lab.models import GMLPBlock

torch.manual_seed(42)
block = GMLPBlock(width=8, expansion=16, max_length=9)

x = torch.randn(2, 9, 8)
changed = x.clone()
changed[:, 5:] += 9.0

mask = torch.tril(torch.ones(9, 9))

before = block(x, mask)[:, :5]
after = block(changed, mask)[:, :5]

assert torch.allclose(before, after, atol=1e-6)
```

Our test suite also checks the opposite condition: without a mask and with a nonzero spatial matrix, changing future tokens **can** change earlier representations.

This is stronger than asserting the code contains a triangular mask. It verifies the behavioral property that the mask is supposed to enforce.

The tests are deterministic and can run on CPU in a few seconds. They are not a formal proof of an entire trained language-model pipeline, but they catch a common implementation error.

## 23. Parameters, FLOPs, and why speed claims need context

The SGU's position matrix has `N x N` learned coefficients for each block, plus `N` learned position biases. Importantly, that matrix is shared across gate channels. The parameter cost of the spatial matrix depends quadratically on configured sequence length, not directly on the channel width.

The spatial multiplication itself processes all gate channels. With `N` tokens and gate width `E/2`, its multiplication cost scales roughly as `N * N * (E/2)`.

Standard full attention also has quadratic token-pair work, along with query, key, value, and output projections. But the architectures perform different operations and offer different kernel-level optimization opportunities.

For our benchmark, the models have different parameter counts:

- gMLP: **15,290**.
- aMLP-style hybrid: **21,626**.
- Transformer attention baseline: **38,978**.

The CPU timing numbers saved in the benchmark are measurements of these specific PyTorch implementations and this very small batch shape, not generally transferable model-efficiency rankings. PyTorch's optimized attention paths, GPU kernels, memory movement, batch size, and compilation can reverse apparent advantages.

The correct performance comparison would record model quality, parameter count, FLOPs, input length, batch size, device, peak memory, preprocessing, end-to-end latency, and generated-token throughput for identical tasks.

We report time only as an implementation diagnostic, not as a claim that one architecture is universally faster.

## 24. Reproduce the study

The companion repository contains the independent code, tests, benchmark configuration, raw results, and figures.

It requires Python and PyTorch. The figures additionally require Matplotlib.

```bash
# Enter the extracted project directory
cd gmlp_spatial_gating_lab

# Install the requirements
python -m pip install -r requirements.txt

# Check all correctness tests
python -m unittest discover -s tests -v

# Inspect SGU tensor shapes and causal masking
python -m gmlp_lab.demo

# Reproduce three-seed experiments and ablations
python -m gmlp_lab.benchmark \
  --seeds 3 \
  --steps 140 \
  --batch-size 96 \
  --eval-size 768 \
  --output results/benchmark.json

# Render figures from authentic saved results
python plot_results.py
```

The baseline configuration runs on CPU and does not download a pretrained checkpoint.

The benchmark uses synthetic data specifically to make the learning problem and its failure modes transparent. It is intentionally not an ImageNet, GLUE, C4, or SQuAD benchmark.

You can also make individual calls to the dataset generator:

```python
from gmlp_lab.data import make_batch

# The marker can appear anywhere.
x, y = make_batch(64, 12, "content", seed=7)

# Train-like marker positions: 1 through 6.
x_train, y_train = make_batch(
    64, 12, "content", seed=8,
    marker_start=1, marker_end=7,
)

# Test-only positions: 7 through 11.
x_test, y_test = make_batch(
    64, 12, "content", seed=9,
    marker_start=7, marker_end=12,
)
```

The marker range is explicit, so you can audit the distribution shift rather than trust a description of it.

## 25. Limitations and what would make the experiment stronger

The biggest limitation is that our study tests **a synthetic rule**, not a general-purpose language model.

The tasks involve twelve tokens and a binary answer. They do not measure syntax, world knowledge, language understanding, or generation quality. The models train from scratch instead of inheriting pretrained representations. They also use different parameter counts.

The strongest conclusions our experiment supports are narrow:

1. The batch-first SGU implementation can learn token mixing and content-sensitive retrieval on this data.
2. Removing cross-token mixing makes these particular query tasks unsolvable for the query-only classifier.
3. A gMLP trained on a restricted set of marker positions can struggle when the same selection rule is applied at new positions.
4. In our setup, a small attention branch and a Transformer baseline transfer the learned selection rule to those held-out positions.
5. Properly masked SGU blocks pass the included prefix-invariance and gradient-causality tests.

The experiment does **not** establish that attention is necessary for extrapolation, that gMLP cannot generalize, or that our aMLP-style model reproduces Google's BERT results.

A stronger research extension would add parameter-matched baselines, a true MLP-Mixer, a Toeplitz-constrained SGU, learned relative positional bias in attention, multiple sequence lengths, variable-length batching, more varied synthetic tasks, and confidence intervals over many more random seeds.

For NLP, a natural next step is to evaluate masked-language modeling and token alignment on a real corpus. You would need to control tokenizer, data, training compute, parameter budget, attention mask, pretraining objective, and downstream fine-tuning hyperparameters before attributing differences to the architecture.

For vision, you could use a small image-classification benchmark and compare a gMLP patch model with a small ViT under matched augmentation and training budgets.

One interesting research question is whether a structured spatial matrix, such as a convolution-like or Toeplitz parameterization, would recover some of the positional transfer that our unconstrained gMLP lacked.

## 26. What gMLP still teaches us

The enduring value of *Pay Attention to MLPs* is not that it predicted the end of self-attention. It did not. Its value is that it dissected a popular architecture and asked which parts were actually responsible for the observed capability.

A model can exchange information among tokens without computing query-key attention weights. A model can achieve good pretraining perplexity without possessing the same downstream transfer bias as an attention model. A tiny amount of adaptive routing may be enough to repair a weakness without abandoning the simpler backbone.

Our experiment places those ideas in twelve token positions.

When every marker position was represented in training, gMLP learned the retrieval rule. When we moved the marker to positions it had never been trained to use, its performance collapsed. Adding a small attention head restored the behavior on the same shift.

No sweeping claim about the superiority of MLPs or Transformers is necessary. The experiment makes a more useful engineering point:

**A network's ability to fit a pattern is not the same as having the right inductive bias to transfer that pattern.**

That is the lesson worth taking from gMLP into the next generation of neural architectures.

---

## 27. References and further reading

**[1] Liu, Hanxiao; Dai, Zihang; So, David R.; Le, Quoc V. (2021).** *Pay Attention to MLPs.* NeurIPS 34. Original paper: https://arxiv.org/abs/2105.08050 . NeurIPS proceedings: https://papers.neurips.cc/paper_files/paper/2021/hash/4cc05b35c2f937c5bd9e7d41d3686fff-Abstract.html

**[2] labml.ai.** *Pay Attention to MLPs (gMLP): Annotated PyTorch Implementation.* https://nn.labml.ai/transformers/gmlp/index.html

**[3] Vaswani et al. (2017).** *Attention Is All You Need.* https://arxiv.org/abs/1706.03762

**[4] Tolstikhin et al. (2021).** *MLP-Mixer: An All-MLP Architecture for Vision.* https://arxiv.org/abs/2105.01601

**[5] Touvron et al. (2021).** *ResMLP: Feedforward Networks for Image Classification with Data-Efficient Training.* https://arxiv.org/abs/2105.03404

**[6] Devlin et al. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* https://arxiv.org/abs/1810.04805

**[7] Dauphin et al. (2017).** *Language Modeling with Gated Convolutional Networks.* https://arxiv.org/abs/1612.08083

**[8] Shazeer (2020).** *GLU Variants Improve Transformer.* https://arxiv.org/abs/2002.05202

**[9] PyTorch Image Models (timm).** Implementations of Spatial Gating Unit and gMLP vision variants. https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py

**[10] Community implementation.** *g-mlp-pytorch* by lucidrains, including optional tiny-attention variants. https://github.com/lucidrains/g-mlp-pytorch

**[11] Independent companion lab for this article.** Original educational PyTorch code, deterministic synthetic data, controlled experiments, 21 tests, and saved results. Distributed separately with the article. Not affiliated with Google Research or labml.ai.

*Disclosure: All independent benchmark results in this article come from a small synthetic CPU experiment. Published ImageNet, BERT, and SQuAD results are attributed to the original paper. The code is educational and is not a reproduction of the full NeurIPS training setup.*
