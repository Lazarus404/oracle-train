# Getting started

## What you need

- Jello: `jelloc` and `jello` on your `PATH`.
- A C compiler, to build the native libraries: `cc` on macOS and Linux, or llvm-mingw's `clang` on Windows (run `build.sh` from Git Bash there).
- ONNX Runtime **with its training API**. The `onnxruntime-training` releases on [ONNX Runtime's releases page](https://github.com/microsoft/onnxruntime/releases) have it, for Linux and Windows, and `build.sh` downloads one when it finds none; on macOS, build it from source with `scripts/build-onnxruntime.sh`. The plain inference releases for macOS and Linux, and Homebrew's `onnxruntime`, do not have it. oracle-train builds against an ONNX Runtime only if it holds `onnxruntime_training_c_api.h`, and then `available()` says whether the library can really train.

## Install

Add oracle-train to the `dependencies` of your project's `jello.json`:

```json
{
  "name": "digits",
  "version": "0.1.0",
  "dependencies": {
    "oracle-train": {"github": "Lazarus404/oracle-train", "tag": "0.1.0"}
  }
}
```

Fetch it, and oracle, which it depends on, then build their native libraries: oracle's as [its guide](https://github.com/Lazarus404/oracle/blob/main/docs/getting-started.md) says, against the ONNX Runtime it needs, and oracle-train's against the training build:

```sh
jelloc deps.get
packages/oracle/build.sh
packages/oracle-train/build.sh
```

Each library loads the ONNX Runtime it was built against, so the two may be different versions.

`build.sh` finds Jello's C headers through `JELLO_ROOT`, if you set it, or else beside the `jello` on your `PATH`. It looks for ONNX Runtime, in order:

1. through `pkg-config`, as `libonnxruntime`;
2. under `ONNXRUNTIME_ROOT`, a directory holding `include/` and `lib/`, such as an `onnxruntime-training` release unpacked;
3. in `third_party/onnxruntime-training`, beside `build.sh`.

It counts each only if it has the training API's header. When none has it, on Linux (x64 and aarch64) and Windows (x64, arm64 and x86) it downloads the `onnxruntime-training` 1.19.2 release for your system and your compiler's architecture, such as `onnxruntime-training-linux-x64-1.19.2.tgz`, with `curl`, checks it against the SHA-256 pinned in `build.sh`, and unpacks it into `third_party/onnxruntime-training`, keeping the archive in `build/`; on Windows, it also needs `unzip`.

Microsoft publishes no training release for macOS, so there `build.sh` stops and points to `scripts/build-onnxruntime.sh`. That clones ONNX Runtime 1.19.2 into `build/`, builds it with its training API for your Mac's architecture, and installs it into `third_party/onnxruntime-training`, laid out as the releases are. It needs `git`, Python 3, Xcode's command line tools and the other tools in [ONNX Runtime's build guide](https://onnxruntime.ai/docs/build/training.html), and takes an hour or more; `--print` shows the commands it would run without running them:

```sh
packages/oracle-train/scripts/build-onnxruntime.sh
```

Set `CC` to choose another compiler. The library lands in `packages/oracle-train/jdll/`, where `jelloc run` and `jelloc test` look for it, and it finds ONNX Runtime's library where it was built against it.

## The artifacts

ONNX Runtime trains from four files, which its Python package `onnxruntime-training` makes from an ONNX model with `onnxruntime.training.artifacts.generate_artifacts`:

| Artifact | Holds |
|----------|-------|
| the checkpoint | the parameters, and the optimizer's state |
| the training model | the model, its loss, and the loss's gradients |
| the eval model | the model and its loss, without gradients |
| the optimizer model | the optimizer, such as AdamW, that updates the parameters |

## A first training run

Put this in `src/Main.jello`, with the artifacts in `artifacts/`:

```jello
import { open } from oracle_train.Trainer;
import { fromF32, fromI32 } from oracle.Tensor;

let trainer = open({
  checkpoint: "artifacts/checkpoint",
  trainingModel: "artifacts/training_model.onnx",
  evalModel: "artifacts/eval_model.onnx",
  optimizerModel: "artifacts/optimizer_model.onnx",
});
let batch = [
  fromF32("input", [2, 4], [0.1f32, 0.9f32, 0.0f32, 0.0f32, 0.9f32, 0.1f32, 0.0f32, 0.0f32]),
  fromI32("labels", [2], [1, 0]),
];
let before = trainer.evalLoss(batch);
let step = 0;
while (step < 20) { trainer.trainStep(batch); step++; }
let after = trainer.evalLoss(batch);
trainer.close();
`the loss fell: ${after < before}`
```

and run it:

```sh
$ jelloc run
the loss fell: true
```

`evalLoss` gives the loss as an `F64`, whose value depends on the model. `open` throws when an artifact will not load, and when the ONNX Runtime underneath has no training API.

## Next

[Training](ort-training.md) covers each step, checkpoints, and exporting the trained model to run with oracle.
