# goGPT: The Compiled Evolution

**A character-level decoder-only Transformer implemented in Go.**

This is the direct evolution of `bashGPT`. While the original was a proof-of-concept for the "unreasonable" use of shell scripting, `goGPT` maintains the same educational transparency conscious tensor libraries for raw scalar math while leveraging Go's concurrency and speed.

## The Evolution: From Bash to Go

| Feature | bashGPT (v2) | goGPT |
| :--- | :--- | :--- |
| **Runtime** | Interpreted (Bash + `bc`) | Compiled (Go 1.22+) |
| **Arithmetic** | Process I/O via `coproc` | Native `float64` |
| **Data Structures** | Associative Arrays | Native Maps & Slices |
| **Improved Performance** | ~11 minutes / run | < 60 seconds / run at 2000 steps |

*NOTE:* In the original bashGPT, we used 100 entries over 100 steps. Here, we use 32,000 entried over 2000 steps and still reach under 1 minute.

## Core Philosophy: No Tensors Allowed

Most modern Transformer implementations are buried under layers of abstraction (PyTorch, TensorFlow etc). `goGPT` is designed for the developer who wants to see the gears turn.

Once the basic math is presented and clear, porting this to any language becomes relatively painless.

[QBasic](https://github.com/DualBrain/QB64) anyone? :innocent:

* **Scalar Autograd:** Every operation (add, mul, exp) is tracked as an individual node in a computation graph.
* **Manual Backprop:** We build a topological sort and walk the gradients back manually.
* **Zero Dependencies:** No ML libraries. Only the Go standard library.

## Technical Specifications

The default "Tiny" configuration is designed for rapid iteration:
- **Architecture:** 1 Layer, 2 Attention Heads.
- **Embeddings:** 8-dimensional.
- **Context:** 8 tokens (characters).
- **Optimizer:** Adam (with bias correction).
- **Normalization:** RMSNorm.

## Getting Started

### Prerequisites
* Go 1.22 or higher.
* `curl` (used internally for initial data fetch).

### Installation & Execution
```bash
git clone https://github.com/ResonantEntropy/goGPT.git
cd goGPT

go run main.go
```

### Configuration
Hyperparameters are defined as constants at the top of `main.go`. You can adjust `nEmbd`, `nLayer`, or `numSteps` to observe how the model capacity impacts learning stability.

## Safety & Performance Note
While significantly faster than its Bash predecessor, this is still a **scalar-based engine**. It is intended for educational inspection of gradients and weights, not for production-grade LLM training. Use at own risk.

## Acknowledgments
- Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore).
- Originally conceived as [bashGPT](https://github.com/ResonantEntropy/bashGPT) to learn how Transformers work and to explore the limits of shell-based logic.
