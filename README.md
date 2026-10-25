# oracle-train

**oracle-train** trains ONNX models from Jello programs. It wraps the training API of [ONNX Runtime](https://onnxruntime.ai) in a `Trainer`: give it the artifacts ONNX Runtime's training tools make from a model, feed it batches of tensors, and it runs training steps, measures the loss, saves checkpoints and exports a trained model for inference. Batches and outputs are the `Tensor`s of [oracle](https://github.com/Lazarus404/oracle), which runs the exported model.

**Version:** 0.1.0

```jello
import { open } from oracle_train.Trainer;
import { fromF32, fromI32 } from oracle.Tensor;

let trainer = open({
  checkpoint: "artifacts/checkpoint",
  trainingModel: "artifacts/training_model.onnx",
  evalModel: "artifacts/eval_model.onnx",
  optimizerModel: "artifacts/optimizer_model.onnx",
});
let batch = [fromF32("input", [1, 4], [0.1f32, 0.9f32, 0.0f32, 0.0f32]), fromI32("labels", [1], [1])];
trainer.trainStep(batch);                          // one step of training
let loss = trainer.evalLoss(batch);                // the loss now
trainer.exportInference("model.onnx", ["output"]); // a model oracle.Session opens
trainer.close();
```

## Install

Add oracle-train to your project's `jello.json`; it brings oracle with it:

```json
"dependencies": {
  "oracle-train": {"github": "Lazarus404/oracle-train", "tag": "0.1.0"}
}
```

Fetch the packages, then build both native libraries, which need a C compiler:

```sh
jelloc deps.get
packages/oracle/build.sh
packages/oracle-train/build.sh
```

oracle's `build.sh` finds or downloads the ONNX Runtime it needs, as its README says. oracle-train needs a build of ONNX Runtime **with its training API**, which the plain inference releases for macOS and Linux, and Homebrew's `onnxruntime`, lack. Its `build.sh` uses one it finds through `pkg-config`, under `ONNXRUNTIME_ROOT` or in `packages/oracle-train/third_party/onnxruntime-training`. When it finds none, on Linux (x64 and aarch64) and Windows (x64, arm64 and x86) it downloads the `onnxruntime-training` 1.19.2 release from [ONNX Runtime's releases](https://github.com/microsoft/onnxruntime/releases), checks the archive against a pinned SHA-256, and unpacks it there. Microsoft publishes no training release for macOS, so there, build ONNX Runtime from source into that directory first, which takes an hour or more:

```sh
packages/oracle-train/scripts/build-onnxruntime.sh
```

Each library loads the ONNX Runtime it was built against, so the two may differ in version.

`jelloc run` and `jelloc test` find the libraries in the packages' `jdll/` directories on their own.

## What you get

| Name | Does |
|------|------|
| `open(paths, ?options)`, `openBytes(contents, ?options)` | a `Trainer` of four training artifacts, as files or as Bytes |
| `trainStep(batch)` | one step of training; the training model's outputs |
| `evalStep(batch)`, `evalLoss(batch)` | the eval model's outputs, or its loss alone |
| `saveCheckpoint(path, ?includeOptimizer)` | the parameters, saved to train on later |
| `exportInference(path, outputs)` | an inference model of the trained parameters |
| `close()` | free the trainer now, rather than when it is collected |
| `available()` | whether the ONNX Runtime underneath has the training API |

They are all in `oracle_train.Trainer`, with the `Artifacts` and `Options` records; `version()` is in `oracle_train.Meta`. Errors a program can do something about, such as an artifact that will not load or a batch missing an input, throw Bytes that say what went wrong.

## Documentation

The [guide](docs/README.md) starts at the beginning and covers everything:

| Guide | Covers |
|-------|--------|
| [Getting started](docs/getting-started.md) | Installing ONNX Runtime's training build and oracle-train, and a first training run |
| [Training](docs/ort-training.md) | Artifacts, batches, steps, the loss, checkpoints and exporting |
| [How it works](docs/architecture.md) | The native library, handles, blocking calls and threads |
| [Reference](docs/modules.md) | Every type and function |
| [Testing](docs/testing.md) | Running the tests, and the fixtures |

## Development

```sh
./build.sh      # builds jdll/oracle_train.jdll
./test.sh       # builds, then runs the unit tests, the examples and the programs
```

Both need Jello's `jelloc` and `jello` on `PATH`. `build.sh` downloads ONNX Runtime's training build into `third_party/onnxruntime-training` when it finds it nowhere else, except on macOS, where `scripts/build-onnxruntime.sh` builds it there from source. Beside oracle, in the same parent directory, oracle-train finds it there, through `"package_paths": [".."]`; build oracle's native library first, with its own `build.sh`. The examples in `examples/` run as tests, so they always work as written.

## Licence

oracle-train is under the BSD 3-Clause licence. ONNX Runtime is Microsoft's, under the MIT licence; the fixtures in `fixtures/training_api/` are from its test data, under the same licence.
