# Reference

```jello
import { open, openBytes, available, defaultOptions } from oracle_train.Trainer;
import type { Trainer, Artifacts, Options } from oracle_train.Trainer;
import { version } from oracle_train.Meta;
```

Functions that take `?options` use `defaultOptions()` when it is left out. Errors a program can handle throw Bytes that start with the function's name, as `Trainer.open: ...`. Batches and outputs are oracle's `Tensor`s, from `oracle.Tensor`.

## oracle_train.Trainer

| Type | What it is |
|------|------------|
| `Trainer` | a model in training in ONNX Runtime; made with `open` or `openBytes` |
| `Artifacts` | `{ checkpoint: Bytes, trainingModel: Bytes, evalModel: Bytes, optimizerModel: Bytes }`: the four artifacts' paths, or their contents |
| `Options` | `{ threads: I32 }`; `threads` 0 lets ONNX Runtime choose |

| Function | Result | Does |
|----------|--------|------|
| `open(paths, ?options)` | `Trainer` | loads the artifacts at the paths in `paths`; throws when one will not load, or ONNX Runtime has no training API |
| `openBytes(contents, ?options)` | `Trainer` | loads artifacts held in Bytes, through a temporary directory the trainer removes when it is freed |
| `available()` | `Bool` | whether the ONNX Runtime underneath has its training API |
| `defaultOptions()` | `Options` | `{ threads: 1 }` |

| Trainer method | Result | Does |
|----------------|--------|------|
| `trainStep(batch)` | `Array<Tensor>` | one step of training: forward, backward, an optimizer update and the gradients reset; the training model's outputs |
| `evalStep(batch)` | `Array<Tensor>` | the eval model's outputs on the parameters as they are |
| `evalLoss(batch)` | `F64` | the first element of the eval model's first output; throws when it holds no `F32` or `F64` |
| `saveCheckpoint(path, ?includeOptimizer)` | `Null` | writes the parameters, and the optimizer's state when `includeOptimizer` is true (false by default), to a checkpoint file |
| `exportInference(path, outputs)` | `Null` | writes an inference model of the eval model and the parameters, giving the outputs named |
| `close()` | `Null` | frees the trainer now; a closed trainer traps when used |

A step takes the tensors of `batch` that its model has inputs of, matched by name, and throws when one of the model's inputs is missing or ONNX Runtime refuses the batch.

## oracle_train.Meta

| Function | Result |
|----------|--------|
| `version()` | `"0.1.0"` |
