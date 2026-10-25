# The oracle-train guide

oracle-train trains ONNX models in ONNX Runtime from Jello. These pages explain it from the beginning; read them in order, or go straight to the one you need.

1. [Getting started](getting-started.md): install ONNX Runtime's training build and oracle-train, and train a model for a few steps.
2. [Training](ort-training.md): the four artifacts, batches, training and eval steps, the loss, checkpoints and exporting a model for inference.
3. [How it works](architecture.md): the native library underneath, handles, blocking calls and threads.
4. [Reference](modules.md): every type and function on one page.
5. [Testing](testing.md): how oracle-train is tested, and how to run the tests.

oracle-train builds on [oracle](https://github.com/Lazarus404/oracle), whose `Tensor` it takes and gives, and whose `Session` runs the models it exports.

Back to the [README](../README.md).
