# How it works

oracle-train has two halves, and leans on oracle for its tensors:

```text
 your program
      |
 oracle_train.Trainer    Jello: the Trainer prototype, Artifacts, Options, and the
      |                         checks that throw on what ONNX Runtime refuses
 oracle_train.jdll       C: the training session, its steps, checkpoints and export
      |
 ONNX Runtime            its training API, in a training build of libonnxruntime
```

## The native library

`oracle_train.jdll` is a Jello native library, built from `jdll/oracle_train.c` by `build.sh` and linked against ONNX Runtime. `oracle_train.Trainer` loads its functions and gives each a Jello type, and the `Trainer` prototype wraps them in methods. A program never sees the natives themselves.

Batches go to the library, and outputs come back, in oracle's [wire format](https://github.com/Lazarus404/oracle/blob/main/docs/bytes.md): the Jello side calls oracle's `encodeAll` and `decodeFrom`, and the library reads and writes the same bytes. An answer that ONNX Runtime may refuse starts with a status, and the Jello side throws the message that follows a refusal.

oracle-train loads ONNX Runtime itself, through its own library: an oracle `Session` and a `Trainer` do not share one. The two libraries may even be built against different versions of ONNX Runtime, as when the training build is older than the one oracle uses; each finds its own.

## Handles and memory

A trainer's native side, an ONNX Runtime training session and the checkpoint state it trains, is a *native handle*, held by the `Trainer` in its `handle` field. To Jello the handle has an abstract type: it can only be passed back to oracle-train's natives.

The garbage collector frees a trainer's native side once nothing holds the `Trainer`. `close()` frees it at once instead, which is worth doing, since a model's parameters, gradients and optimizer state can be large. Using a trainer after closing it is a bug, and the program stops with the trap `TYPE_MISMATCH`; closing it twice does nothing. A trainer from `openBytes` also removes its temporary directory when it is freed.

## Blocking calls and threads

Opening a trainer, its steps, and saving and exporting all wait on ONNX Runtime. They are *blocking* natives: the process that calls one parks while a thread of the VM's blocking pool does the work, and the other processes run meanwhile. The threads ONNX Runtime uses within a step are its own, as many as `Options.threads` allows.

An ONNX Runtime training session is not safe to use from two threads at once, so a trainer belongs to the process that opened it. Sending one to another process, or capturing one in a function another process runs, stops the program with the trap `UNSENDABLE`. To train in several processes, open a trainer in each.
