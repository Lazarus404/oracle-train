# Training

Everything here is in `oracle_train.Trainer`:

```jello
import { open, openBytes, available, defaultOptions } from oracle_train.Trainer;
import type { Trainer, Artifacts, Options } from oracle_train.Trainer;
```

## Opening a trainer

`open(paths, ?options)` loads the four artifacts (see [Getting started](getting-started.md#the-artifacts)) from files and gives a `Trainer`. `paths` is an `Artifacts` record:

```jello
let trainer = open({
  checkpoint: "artifacts/checkpoint",
  trainingModel: "artifacts/training_model.onnx",
  evalModel: "artifacts/eval_model.onnx",
  optimizerModel: "artifacts/optimizer_model.onnx",
}, { threads: 4 });
```

`options` is an `Options` record, `{ threads: I32 }`: how many threads one step may use, 0 leaving the choice to ONNX Runtime. Without it, a trainer uses `defaultOptions()`, one thread. A negative count throws.

`openBytes(contents, ?options)` takes the artifacts' contents instead, in the same record, for artifacts that come from somewhere other than files, such as a download. It writes them to a temporary directory of the trainer's own, since ONNX Runtime reads the eval model's file again to export an inference model, and removes it when the trainer is closed or collected.

Both throw Bytes such as `Trainer.open: cannot load the checkpoint: ...` when an artifact cannot be read or loaded, and `Trainer.open: this ONNX Runtime has no training API; ...` when the ONNX Runtime underneath cannot train. `available()` says which, without opening anything:

```jello
if (!available()) { throw "this program needs ONNX Runtime's training build"; }
```

## Batches

A batch is an array of oracle `Tensor`s, named as the models' inputs are named. ONNX Runtime's training and eval models usually take the model's own inputs and the labels the loss compares its outputs with:

```jello
import { fromF32, fromI32 } from oracle.Tensor;

let batch = [
  fromF32("input", [2, 4], [0.1f32, 0.9f32, 0.0f32, 0.0f32, 0.9f32, 0.1f32, 0.0f32, 0.0f32]),
  fromI32("labels", [2], [1, 0]),
];
```

Each step matches the batch to its model's inputs by name, in any order, and leaves out tensors the model does not take, so one batch serves both models. A batch missing one of the model's inputs throws, as `Trainer.trainStep: the input labels is missing`. The element types are oracle's: `F32`, `F64`, `I32`, `I64` and `U8`.

## Training and evaluating

| Method | Does |
|--------|------|
| `trainStep(batch)` | runs the training model forward and back, lets the optimizer update the parameters, and resets the gradients; gives the training model's outputs |
| `evalStep(batch)` | runs the eval model on the parameters as they are; gives its outputs |
| `evalLoss(batch)` | the first element of the eval model's first output, as an `F64` |

The outputs are oracle `Tensor`s; for the models `onnxruntime-training` makes, the loss comes first. `evalLoss` is the usual way to watch training:

```jello
let step = 0;
while (step < 100 && trainer.evalLoss(batch) > 0.1) {
  trainer.trainStep(batch);
  step++;
}
```

A step that ONNX Runtime refuses, as for a tensor of the wrong shape or type, throws `Trainer.trainStep: the training step failed: ...` with ONNX Runtime's message; `evalLoss` also throws when the first output holds no `F32` or `F64`.

## Checkpoints

`saveCheckpoint(path, ?includeOptimizer)` writes the parameters as they now are to a checkpoint file. `includeOptimizer`, false by default, saves the optimizer's state too, such as AdamW's moments, so that training resumes exactly where it stopped. To go on training later, open the saved checkpoint with the same models:

```jello
trainer.saveCheckpoint("artifacts/step-100");
trainer.close();

let resumed = open({
  checkpoint: "artifacts/step-100",
  trainingModel: "artifacts/training_model.onnx",
  evalModel: "artifacts/eval_model.onnx",
  optimizerModel: "artifacts/optimizer_model.onnx",
});
```

## Exporting a model for inference

`exportInference(path, outputs)` writes an ordinary ONNX model to `path`: the eval model with the trained parameters, cut down to give only the outputs named in `outputs`, so the loss and the labels it needs go. oracle runs it like any other model:

```jello
import { open as openSession } from oracle.Session;

trainer.exportInference("model.onnx", ["output"]);
let session = openSession("model.onnx");
let scores = session.runOne(batch[0]);
session.close();
```

An output the eval model does not have, or a path that cannot be written, throws `Trainer.exportInference: cannot export the inference model: ...`.

## Closing

`close()` frees a trainer's session and parameters at once; a trainer nobody holds is freed when the garbage collector finds it. Closing twice does nothing, and using a closed trainer is a bug that stops the program with the trap `TYPE_MISMATCH` (see [How it works](architecture.md)).
