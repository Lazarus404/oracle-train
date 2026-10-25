# Testing

```sh
./test.sh
```

`test.sh` builds `jdll/oracle_train.jdll`, fetches the packages if `packages/` lacks degust, the test dependency, and runs `jelloc test`. oracle comes from `packages/oracle`, or else from `../oracle`, beside this package; build its library first with its own `build.sh`, since it may need another ONNX Runtime than the training build this package's `build.sh` finds or downloads, as [Getting started](getting-started.md) says. That compiles every test into one binary and runs:

- **the unit tests**, in `tests/oracle_train/Trainer.test.jello`, on `std.testing`: opening, training until the loss falls to what ONNX Runtime's own tests expect, eval steps, saving and resuming a checkpoint, exporting a model and running it with oracle, opening from Bytes, and the errors a program can catch;
- **the examples**, every program in `examples/`, which [degust](https://github.com/Lazarus404/degust) compiles and runs, checking each prints what its `// expect:` lines say, with the JIT and without;
- **the programs** in `tests/programs/`, for what a unit test cannot check, such as a closed trainer stopping the program with a trap.

The tests that train, the examples and the programs need ONNX Runtime's training API. Where the ONNX Runtime underneath lacks it, they show as `SKIP`, and a test checks that opening a trainer throws the error saying so; the rest still run.

Arguments to `test.sh` go to the test binary:

```sh
./test.sh -v                          # every test's result, not only failures
./test.sh --run oracle_train/Trainer  # only the unit tests
./test.sh --run Programs              # only the examples and programs
```

## The fixtures

`fixtures/training_api/` holds ONNX Runtime's own test artifacts for its training API: a small classifier of 784 inputs and 10 classes, with its checkpoint (`checkpoint.ckpt`), training, eval and optimizer models (`training_model.onnx`, `eval_model.onnx`, `adamw.onnx`), one batch of two rows (`input-0.in`, `labels.in`), and the eval losses before and after training on it (`loss_1.out`, `loss_2.out`). Each of those text files names its tensor on its first line and lists the numbers on its second. `tests/oracle_train/Fixture.jello` reads them into oracle tensors.
