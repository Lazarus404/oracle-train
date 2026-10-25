/* SPDX-License-Identifier: BSD-3-Clause */
/**
 * @file oracle_train.c
 * @brief ONNX Runtime training for Jello: train steps, eval steps, checkpoints and inference
 * export, through ONNX Runtime's training C API.
 *
 * A trainer is a handle whose payload holds an ONNX Runtime training session and the checkpoint
 * state it trains; it stays in the process that opened it, so its class has no `retain`.
 * Opening, stepping and saving are blocking exports: they read and write files and do the
 * training on a thread of the VM's blocking pool.
 *
 * Tensors cross the border in oracle's wire format, used for a step's inputs and its outputs
 * alike. All numbers are little-endian, and every field starts at a multiple of 4 bytes:
 *
 *   u32 count                      the number of tensors, then for each:
 *   u32 name_len, name             the name's bytes, then zero bytes to a multiple of 4
 *   u32 dtype                      DTYPE_*
 *   u32 rank, i32 dims[rank]       the shape
 *   u32 data_len, data             the elements, packed, then zero bytes to a multiple of 4
 *
 * An export that ONNX Runtime may refuse answers Bytes that start with a u32 status: STATUS_OK
 * and its result, or STATUS_ERROR and the message saying why, for the Jello side to throw.
 */

#include <jello/jdll.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_training_c_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/** Element types, as the wire format and oracle.Tensor number them; 0 is any other. */
enum {
  DTYPE_OTHER = 0,
  DTYPE_F32 = 1,
  DTYPE_F64 = 2,
  DTYPE_I32 = 3,
  DTYPE_I64 = 4,
  DTYPE_U8 = 5,
};

/** The status that starts the answer of an export ONNX Runtime may refuse. */
enum { STATUS_OK = 0, STATUS_ERROR = 1 };

/** The models a step runs. */
enum { MODEL_TRAINING, MODEL_EVAL };

/* ---------------------------------------------------------------------------------------------
 * ONNX Runtime, started once per process
 * ------------------------------------------------------------------------------------------- */

static const OrtApi* ort;
static const OrtTrainingApi* training;
static OrtEnv* env;
static char start_error[256];

/**
 * Finds the API, its training part and makes the one environment; on failure, says why in
 * start_error. ONNX Runtime logs only fatal errors: every other error comes back to the caller.
 */
static void start(void) {
  const OrtApiBase* base = OrtGetApiBase();
  ort = base ? base->GetApi(ORT_API_VERSION) : NULL;
  if (!ort) {
    snprintf(start_error, sizeof start_error,
             "ONNX Runtime %s does not offer the API this library was built for",
             base ? base->GetVersionString() : "(unknown)");
    return;
  }
  OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "oracle_train", &env);
  if (status) {
    snprintf(start_error, sizeof start_error, "cannot start ONNX Runtime: %s",
             ort->GetErrorMessage(status));
    ort->ReleaseStatus(status);
    env = NULL;
    return;
  }
  training = ort->GetTrainingApi(ORT_API_VERSION);
}

#if defined(_WIN32)
static INIT_ONCE started = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK start_once(PINIT_ONCE once, PVOID param, PVOID* context) {
  (void)once;
  (void)param;
  (void)context;
  start();
  return TRUE;
}
#else
static pthread_once_t started = PTHREAD_ONCE_INIT;
#endif

/** Starts ONNX Runtime if no call has yet; 0, failing the call, when it cannot start. */
static int ready(jdlo_ctx* c) {
#if defined(_WIN32)
  InitOnceExecuteOnce(&started, start_once, NULL, NULL);
#else
  pthread_once(&started, start);
#endif
  if (!env) {
    jdl_fail(c, start_error);
    return 0;
  }
  return 1;
}

/* ---------------------------------------------------------------------------------------------
 * Little-endian numbers and a growing buffer
 * ------------------------------------------------------------------------------------------- */

static uint32_t read_u32(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/** Bytes written in order, which grow as needed; `failed` once memory runs out. */
typedef struct buffer {
  uint8_t* data;
  size_t len;
  size_t cap;
  int failed;
} buffer;

/** Room for N more bytes; NULL once memory has run out. */
static uint8_t* buffer_grow(buffer* b, size_t n) {
  if (b->failed) return NULL;
  if (b->len + n > b->cap) {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + n) cap *= 2;
    uint8_t* data = realloc(b->data, cap);
    if (!data) {
      b->failed = 1;
      return NULL;
    }
    b->data = data;
    b->cap = cap;
  }
  uint8_t* at = b->data + b->len;
  b->len += n;
  return at;
}

static void put_u32(buffer* b, uint32_t v) {
  uint8_t* p = buffer_grow(b, 4);
  if (!p) return;
  for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static void put_bytes(buffer* b, const void* data, size_t len) {
  uint8_t* p = buffer_grow(b, len);
  if (p && len) memcpy(p, data, len);
}

/** Zero bytes up to the next multiple of 4. */
static void put_pad(buffer* b) {
  static const uint8_t zeros[4] = {0};
  put_bytes(b, zeros, (4 - b->len % 4) % 4);
}

/** Returns the buffer as Bytes, failing the call when memory ran out or it is too long. */
static void return_buffer(jdlo_ctx* c, buffer* b) {
  if (b->failed) {
    jdl_fail(c, "out of memory");
  } else if (b->len > UINT32_MAX) {
    jdl_fail(c, "the answer is over 4 GiB");
  } else {
    jdl_return_bytes_copy(c, b->data, (uint32_t)b->len);
  }
  free(b->data);
}

/** Returns STATUS_OK and nothing more. */
static void return_ok(jdlo_ctx* c) {
  buffer b = {0};
  put_u32(&b, STATUS_OK);
  return_buffer(c, &b);
}

/** Returns STATUS_ERROR and MESSAGE, for the Jello side to throw. */
static void return_refusal(jdlo_ctx* c, const char* message) {
  buffer b = {0};
  put_u32(&b, STATUS_ERROR);
  put_bytes(&b, message, strlen(message));
  return_buffer(c, &b);
}

/** Returns STATUS_ERROR and STATUS's message after WHAT; frees STATUS. */
static void return_status(jdlo_ctx* c, const char* what, OrtStatus* status) {
  char text[1024];
  snprintf(text, sizeof text, "%s: %s", what, ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return_refusal(c, text);
}

/** Returns MESSAGE as Bytes, the answer of an open that failed, for the Jello side to throw. */
static void return_message(jdlo_ctx* c, const char* message) {
  jdl_return_bytes_copy(c, (const uint8_t*)message, (uint32_t)strlen(message));
}

/** Whether STATUS is success; frees it when it is not. */
static int succeeded(OrtStatus* status) {
  if (!status) return 1;
  ort->ReleaseStatus(status);
  return 0;
}

/* ---------------------------------------------------------------------------------------------
 * Paths and files
 * ------------------------------------------------------------------------------------------- */

/** A NUL-terminated copy of LEN bytes at DATA; NULL when memory runs out. */
static char* copy_text(const uint8_t* data, size_t len) {
  char* text = malloc(len + 1);
  if (text) {
    if (len) memcpy(text, data, len);
    text[len] = 0;
  }
  return text;
}

/** Path argument INDEX as a C string; NULL for an empty one or one holding a zero byte. */
static char* arg_path(jdlo_ctx* c, int index) {
  const uint8_t* data = jdl_arg_bytes_data(c, index);
  uint32_t len = jdl_arg_bytes_len(c, index);
  if (len == 0 || memchr(data, 0, len)) return NULL;
  return copy_text(data, len);
}

/** PATH as ONNX Runtime takes paths, which the caller frees: UTF-16 on Windows; NULL if it fails. */
static ORTCHAR_T* ort_path(const char* path) {
#if defined(_WIN32)
  int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
  wchar_t* wide = len > 0 ? malloc((size_t)len * sizeof *wide) : NULL;
  if (wide) MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, len);
  return wide;
#else
  return copy_text((const uint8_t*)path, strlen(path));
#endif
}

/** The size of the file at PATH, for the collector's estimate; 0 when it cannot be read. */
static uint32_t file_size(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return 0;
  long size = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : 0;
  fclose(f);
  return size <= 0 ? 0 : size > (long)UINT32_MAX ? UINT32_MAX : (uint32_t)size;
}

/** DIR joined to NAME, which the caller frees; NULL when memory runs out. */
static char* join(const char* dir, const char* name) {
  size_t len = strlen(dir) + 1 + strlen(name);
  char* path = malloc(len + 1);
  if (path) snprintf(path, len + 1, "%s/%s", dir, name);
  return path;
}

/** Writes LEN bytes at DATA to the file at DIR/NAME; whether it succeeded. */
static int write_file(const char* dir, const char* name, const uint8_t* data, size_t len) {
  char* path = join(dir, name);
  FILE* f = path ? fopen(path, "wb") : NULL;
  free(path);
  if (!f) return 0;
  int ok = fwrite(data, 1, len, f) == len;
  return fclose(f) == 0 && ok;
}

/** The files of a session opened from Bytes, in the order its open takes them. */
static const char* const ARTIFACT_FILES[4] = {"checkpoint", "training_model.onnx",
                                              "eval_model.onnx", "optimizer_model.onnx"};

/** A new, empty directory for a session's files, which the caller frees; NULL if it fails. */
static char* make_temp_dir(void) {
#if defined(_WIN32)
  char base[MAX_PATH], name[MAX_PATH];
  if (!GetTempPathA(sizeof base, base) || !GetTempFileNameA(base, "otr", 0, name)) return NULL;
  /* GetTempFileNameA makes a file of the unique name; a directory takes its place. */
  DeleteFileA(name);
  if (!CreateDirectoryA(name, NULL)) return NULL;
  return copy_text((const uint8_t*)name, strlen(name));
#else
  const char* base = getenv("TMPDIR");
  char* pattern = join(base && *base ? base : "/tmp", "oracle_train.XXXXXX");
  if (pattern && !mkdtemp(pattern)) {
    free(pattern);
    return NULL;
  }
  return pattern;
#endif
}

/** Removes a session's files and the directory holding them. */
static void remove_temp_dir(const char* dir) {
  for (int i = 0; i < 4; i++) {
    char* path = join(dir, ARTIFACT_FILES[i]);
    if (path) remove(path);
    free(path);
  }
#if defined(_WIN32)
  _rmdir(dir);
#else
  rmdir(dir);
#endif
}

/* ---------------------------------------------------------------------------------------------
 * Trainers
 * ------------------------------------------------------------------------------------------- */

/**
 * A training session and the checkpoint state it trains, which must outlive it; both are NULL
 * once the trainer is closed. `temp_dir`, for a trainer opened from Bytes, holds its files.
 */
typedef struct trainer {
  OrtTrainingSession* session;
  OrtCheckpointState* checkpoint;
  char* temp_dir;
} trainer;

/** Frees the session, then its checkpoint and its files, leaving the trainer closed. */
static void trainer_release(trainer* t) {
  if (t->session) training->ReleaseTrainingSession(t->session);
  if (t->checkpoint) training->ReleaseCheckpointState(t->checkpoint);
  t->session = NULL;
  t->checkpoint = NULL;
  if (t->temp_dir) {
    remove_temp_dir(t->temp_dir);
    free(t->temp_dir);
    t->temp_dir = NULL;
  }
}

/** Frees a trainer's payload, when no handle holds it any more. */
static void trainer_finalize(void* payload) {
  trainer* t = payload;
  trainer_release(t);
  free(t);
}

static const jello_jdll_handle_class TRAINER = {"oracle_train trainer", trainer_finalize, NULL};

/** The open trainer of argument 0, or NULL once the call has failed. */
static trainer* arg_trainer(jdlo_ctx* c) {
  trainer* t = jdl_arg_handle(c, 0, &TRAINER);
  if (t && !t->session) {
    jdl_fail(c, "the trainer is closed");
    return NULL;
  }
  return t;
}

/** Whether the ONNX Runtime this library runs on has the training API. */
static void training_available(jdlo_ctx* c) {
  if (ready(c)) jdl_return_bool(c, training != NULL);
}

/**
 * Opens a trainer of the files at four paths, the checkpoint and the training, eval and
 * optimizer models, with THREADS threads; returns it as a handle, or Bytes saying why it will not
 * open. Takes TEMP_DIR, the directory of the files, which the trainer removes when it is freed.
 */
static void open_paths(jdlo_ctx* c, char* const paths[4], int32_t threads, char* temp_dir) {
  OrtCheckpointState* checkpoint = NULL;
  OrtSessionOptions* options = NULL;
  OrtTrainingSession* session = NULL;
  ORTCHAR_T* wide[4] = {NULL};
  const char* refused = NULL;
  OrtStatus* status = NULL;
  for (int i = 0; i < 4; i++) {
    if (!(wide[i] = ort_path(paths[i]))) refused = "a path is not UTF-8";
  }
  if (!refused && (status = training->LoadCheckpoint(wide[0], &checkpoint))) {
    refused = "cannot load the checkpoint";
  }
  if (!refused) {
    status = ort->CreateSessionOptions(&options);
    if (!status) status = ort->SetIntraOpNumThreads(options, threads < 0 ? 0 : threads);
    if (!status) {
      status = training->CreateTrainingSession(env, options, checkpoint, wide[1], wide[2],
                                                 wide[3], &session);
    }
    if (status) refused = "cannot load the models";
  }
  if (options) ort->ReleaseSessionOptions(options);
  for (int i = 0; i < 4; i++) free(wide[i]);
  trainer* t = refused ? NULL : malloc(sizeof *t);
  if (!t) {
    if (session) training->ReleaseTrainingSession(session);
    if (checkpoint) training->ReleaseCheckpointState(checkpoint);
    if (temp_dir) {
      remove_temp_dir(temp_dir);
      free(temp_dir);
    }
    if (!refused) {
      jdl_fail(c, "out of memory");
    } else if (status) {
      char text[1024];
      snprintf(text, sizeof text, "%s: %s", refused, ort->GetErrorMessage(status));
      ort->ReleaseStatus(status);
      return_message(c, text);
    } else {
      return_message(c, refused);
    }
    return;
  }
  t->session = session;
  t->checkpoint = checkpoint;
  t->temp_dir = temp_dir;
  uint32_t size = (uint32_t)sizeof *t;
  for (int i = 0; i < 4; i++) {
    uint32_t more = file_size(paths[i]);
    size = more > UINT32_MAX - size ? UINT32_MAX : size + more;
  }
  jdl_return_handle(c, &TRAINER, t, size);
}

/** Says why training cannot start here, as an open's answer; 0 when it can. */
static int refuse_without_training(jdlo_ctx* c) {
  if (training) return 0;
  return_message(c, "this ONNX Runtime has no training API; it must be a build of "
                    "onnxruntime-training");
  return 1;
}

/**
 * Opens a trainer: arguments are the paths of the checkpoint and of the training, eval and
 * optimizer models, and the number of threads. A trainer, or Bytes saying why it will not open.
 */
static void trainer_open(jdlo_ctx* c) {
  if (!ready(c) || refuse_without_training(c)) return;
  char* paths[4];
  int ok = 1;
  for (int i = 0; i < 4; i++) ok &= (paths[i] = arg_path(c, i)) != NULL;
  if (ok) {
    open_paths(c, paths, jdl_arg_i32(c, 4), NULL);
  } else {
    return_message(c, "a path is empty or holds a zero byte");
  }
  for (int i = 0; i < 4; i++) free(paths[i]);
}

/**
 * Opens a trainer of the checkpoint and the training, eval and optimizer models held in Bytes,
 * and the number of threads. They are written to a directory of their own, since ONNX Runtime
 * reads the eval model again to export an inference model, and removed when the trainer is freed.
 */
static void trainer_open_bytes(jdlo_ctx* c) {
  if (!ready(c) || refuse_without_training(c)) return;
  char* dir = make_temp_dir();
  if (!dir) {
    return_message(c, "cannot make a temporary directory for the artifacts");
    return;
  }
  char* paths[4] = {NULL};
  int ok = 1;
  for (int i = 0; i < 4; i++) {
    ok = ok && write_file(dir, ARTIFACT_FILES[i], jdl_arg_bytes_data(c, i), jdl_arg_bytes_len(c, i)) &&
         (paths[i] = join(dir, ARTIFACT_FILES[i])) != NULL;
  }
  if (ok) {
    open_paths(c, paths, jdl_arg_i32(c, 4), dir);
  } else {
    remove_temp_dir(dir);
    free(dir);
    return_message(c, "cannot write the artifacts to a temporary directory");
  }
  for (int i = 0; i < 4; i++) free(paths[i]);
}

/** Frees a trainer now, rather than when its handle is collected. */
static void trainer_close(jdlo_ctx* c) {
  trainer* t = jdl_arg_handle(c, 0, &TRAINER);
  if (t) trainer_release(t);
}

/* ---------------------------------------------------------------------------------------------
 * Steps
 * ------------------------------------------------------------------------------------------- */

static ONNXTensorElementDataType ort_type(uint32_t dtype) {
  switch (dtype) {
    case DTYPE_F32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DTYPE_F64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DTYPE_I32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DTYPE_I64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DTYPE_U8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

static uint32_t wire_type(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DTYPE_F32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return DTYPE_F64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DTYPE_I32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DTYPE_I64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DTYPE_U8;
    default: return DTYPE_OTHER;
  }
}

/** The bytes of one element; 0 for a type the wire format does not carry. */
static size_t element_size(uint32_t dtype) {
  switch (dtype) {
    case DTYPE_F32: case DTYPE_I32: return 4;
    case DTYPE_F64: case DTYPE_I64: return 8;
    case DTYPE_U8: return 1;
    default: return 0;
  }
}

/** The tensors of a step: their names, NUL-terminated, and values, which it owns. */
typedef struct tensors {
  size_t count;
  char** names;
  OrtValue** values;
} tensors;

static int tensors_make(tensors* t, size_t count) {
  t->count = count;
  t->names = calloc(count ? count : 1, sizeof *t->names);
  t->values = calloc(count ? count : 1, sizeof *t->values);
  return t->names && t->values;
}

static void tensors_free(tensors* t) {
  for (size_t i = 0; t->names && t->values && i < t->count; i++) {
    free(t->names[i]);
    if (t->values[i]) ort->ReleaseValue(t->values[i]);
  }
  free(t->names);
  free(t->values);
}

/**
 * Reads a step's inputs, in the wire format, into tensors that ONNX Runtime owns the memory of.
 * 0, having failed the call, when the request is malformed, which only a bug makes it; 0, having
 * returned the reason, when ONNX Runtime refuses a tensor.
 */
static int read_inputs(jdlo_ctx* c, const uint8_t* req, size_t len, OrtAllocator* alloc,
                       tensors* in) {
  size_t count = len < 4 ? 0 : read_u32(req), at = 4, i = 0;
  if (len < 4 || count > len / 4) {
    jdl_fail(c, "the step's inputs are malformed");
    return 0;
  }
  if (!tensors_make(in, count)) {
    jdl_fail(c, "out of memory");
    return 0;
  }
  for (; i < count; i++) {
    if (at + 4 > len) break;
    size_t name_len = read_u32(req + at);
    at += 4;
    if (name_len > len - at) break;
    if (!(in->names[i] = copy_text(req + at, name_len))) {
      jdl_fail(c, "out of memory");
      return 0;
    }
    at += (name_len + 3) / 4 * 4;
    if (at + 8 > len) break;
    uint32_t dtype = read_u32(req + at), rank = read_u32(req + at + 4);
    at += 8;
    if (element_size(dtype) == 0 || rank > 64 || at + 4 * (size_t)rank + 4 > len) break;
    int64_t dims[64];
    size_t elements = 1;
    int bad = 0;
    for (uint32_t d = 0; d < rank; d++) {
      int32_t dim = (int32_t)read_u32(req + at + 4 * d);
      bad |= dim < 0 || (dim > 0 && elements > SIZE_MAX / (size_t)dim);
      dims[d] = dim;
      if (!bad) elements *= (size_t)dim;
    }
    at += 4 * (size_t)rank;
    size_t data_len = read_u32(req + at);
    at += 4;
    if (bad || data_len > len - at || data_len != elements * element_size(dtype)) break;
    OrtStatus* status = ort->CreateTensorAsOrtValue(alloc, dims, rank, ort_type(dtype),
                                                    &in->values[i]);
    void* data = NULL;
    if (!status) status = ort->GetTensorMutableData(in->values[i], &data);
    if (status) {
      return_status(c, "cannot make an input tensor", status);
      return 0;
    }
    if (data_len) memcpy(data, req + at, data_len);
    at += (data_len + 3) / 4 * 4;
  }
  if (i < count || at != len) {
    jdl_fail(c, "the step's inputs are malformed");
    return 0;
  }
  return 1;
}

/**
 * The inputs a model of a session takes, in its order, from IN, matched by name, and the names
 * of its outputs, into *ORDERED, which the caller frees and whose values IN still owns, and OUT.
 * 0, having returned the reason, when an input is missing or ONNX Runtime fails.
 */
static int plan_step(jdlo_ctx* c, OrtTrainingSession* session, int model, OrtAllocator* alloc,
                     const tensors* in, const OrtValue*** ordered, size_t* ordered_count,
                     tensors* out) {
  size_t inputs = 0, outputs = 0;
  OrtStatus* status = model == MODEL_TRAINING
                          ? training->TrainingSessionGetTrainingModelInputCount(session, &inputs)
                          : training->TrainingSessionGetEvalModelInputCount(session, &inputs);
  if (!status) {
    status = model == MODEL_TRAINING
                 ? training->TrainingSessionGetTrainingModelOutputCount(session, &outputs)
                 : training->TrainingSessionGetEvalModelOutputCount(session, &outputs);
  }
  if (status) {
    return_status(c, "cannot read the model's inputs and outputs", status);
    return 0;
  }
  if (!(*ordered = calloc(inputs ? inputs : 1, sizeof **ordered))) {
    jdl_fail(c, "out of memory");
    return 0;
  }
  *ordered_count = inputs;
  for (size_t i = 0; i < inputs; i++) {
    char* name = NULL;
    status = model == MODEL_TRAINING
                 ? training->TrainingSessionGetTrainingModelInputName(session, i, alloc, &name)
                 : training->TrainingSessionGetEvalModelInputName(session, i, alloc, &name);
    if (status) {
      return_status(c, "cannot read the model's inputs", status);
      return 0;
    }
    for (size_t j = 0; j < in->count && !(*ordered)[i]; j++) {
      if (strcmp(in->names[j], name) == 0) (*ordered)[i] = in->values[j];
    }
    if (!(*ordered)[i]) {
      char text[512];
      snprintf(text, sizeof text, "the input %s is missing", name);
      alloc->Free(alloc, name);
      return_refusal(c, text);
      return 0;
    }
    alloc->Free(alloc, name);
  }
  if (!tensors_make(out, outputs)) {
    jdl_fail(c, "out of memory");
    return 0;
  }
  for (size_t i = 0; i < outputs; i++) {
    char* name = NULL;
    status = model == MODEL_TRAINING
                 ? training->TrainingSessionGetTrainingModelOutputName(session, i, alloc, &name)
                 : training->TrainingSessionGetEvalModelOutputName(session, i, alloc, &name);
    if (status) {
      return_status(c, "cannot read the model's outputs", status);
      return 0;
    }
    out->names[i] = copy_text((const uint8_t*)name, strlen(name));
    alloc->Free(alloc, name);
    if (!out->names[i]) {
      jdl_fail(c, "out of memory");
      return 0;
    }
  }
  return 1;
}

/** Puts output tensor VALUE named NAME in the wire format; a message when it cannot be read. */
static const char* put_output(buffer* b, const char* name, OrtValue* value) {
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtStatus* status = ort->GetTensorTypeAndShape(value, &info);
  if (status) {
    ort->ReleaseStatus(status);
    return "an output is not a tensor, which oracle_train cannot read";
  }
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t rank = 0, elements = 0;
  int64_t dims[64];
  void* data = NULL;
  int ok = succeeded(ort->GetTensorElementType(info, &type)) &&
           succeeded(ort->GetDimensionsCount(info, &rank)) && rank <= 64 &&
           succeeded(ort->GetDimensions(info, dims, rank)) &&
           succeeded(ort->GetTensorShapeElementCount(info, &elements));
  ort->ReleaseTensorTypeAndShapeInfo(info);
  uint32_t dtype = wire_type(type);
  if (!ok) return "cannot read an output tensor's shape";
  if (dtype == DTYPE_OTHER) return "an output has an element type oracle_train cannot read";
  if (elements && !succeeded(ort->GetTensorMutableData(value, &data))) {
    return "cannot read an output tensor";
  }
  size_t name_len = strlen(name), data_len = elements * element_size(dtype);
  if (data_len > UINT32_MAX) return "an output is over 4 GiB";
  put_u32(b, (uint32_t)name_len);
  put_bytes(b, name, name_len);
  put_pad(b);
  put_u32(b, dtype);
  put_u32(b, (uint32_t)rank);
  for (size_t d = 0; d < rank; d++) {
    put_u32(b, (uint32_t)(dims[d] < 0 ? -1 : dims[d] > INT32_MAX ? INT32_MAX : (int32_t)dims[d]));
  }
  put_u32(b, (uint32_t)data_len);
  put_bytes(b, data, data_len);
  put_pad(b);
  return NULL;
}

/**
 * Runs a step of a trainer's training or eval model on argument 1, its inputs in the wire
 * format, matched to the model's by name; those it does not take are left out. A training step
 * also updates the parameters and resets the gradients. The answer is STATUS_OK and the model's
 * outputs in the wire format, or STATUS_ERROR and why ONNX Runtime refused the inputs.
 */
static void step(jdlo_ctx* c, int model) {
  trainer* t = arg_trainer(c);
  if (!t) return;
  OrtAllocator* alloc = NULL;
  OrtStatus* status = ort->GetAllocatorWithDefaultOptions(&alloc);
  if (status) {
    ort->ReleaseStatus(status);
    jdl_fail(c, "cannot find ONNX Runtime's allocator");
    return;
  }
  tensors in = {0}, out = {0};
  const OrtValue** ordered = NULL;
  size_t ordered_count = 0;
  if (!read_inputs(c, jdl_arg_bytes_data(c, 1), jdl_arg_bytes_len(c, 1), alloc, &in)) {
    tensors_free(&in);
    return;
  }
  if (!plan_step(c, t->session, model, alloc, &in, &ordered, &ordered_count, &out)) {
    free(ordered);
    tensors_free(&in);
    tensors_free(&out);
    return;
  }
  if (model == MODEL_TRAINING) {
    status = training->TrainStep(t->session, NULL, ordered_count, ordered, out.count, out.values);
    if (!status) status = training->OptimizerStep(t->session, NULL);
    if (!status) status = training->LazyResetGrad(t->session);
  } else {
    status = training->EvalStep(t->session, NULL, ordered_count, ordered, out.count, out.values);
  }
  free(ordered);
  tensors_free(&in);
  if (status) {
    tensors_free(&out);
    return_status(c, model == MODEL_TRAINING ? "the training step failed" : "the eval step failed",
                  status);
    return;
  }
  buffer b = {0};
  put_u32(&b, STATUS_OK);
  put_u32(&b, (uint32_t)out.count);
  const char* refused = NULL;
  for (size_t i = 0; !refused && i < out.count; i++) {
    refused = put_output(&b, out.names[i], out.values[i]);
  }
  tensors_free(&out);
  if (refused) {
    free(b.data);
    return_refusal(c, refused);
    return;
  }
  return_buffer(c, &b);
}

/** A training step: forward, backward and an optimizer step, then the gradients reset. */
static void trainer_train_step(jdlo_ctx* c) { step(c, MODEL_TRAINING); }

/** An eval step: the eval model run on the trainer's current parameters. */
static void trainer_eval_step(jdlo_ctx* c) { step(c, MODEL_EVAL); }

/**
 * Saves a trainer's checkpoint state: arguments are the trainer, the path, and whether to save
 * the optimizer's state too. STATUS_OK, or STATUS_ERROR and why it could not be saved.
 */
static void trainer_save_checkpoint(jdlo_ctx* c) {
  trainer* t = arg_trainer(c);
  if (!t) return;
  char* path = arg_path(c, 1);
  ORTCHAR_T* wide = path ? ort_path(path) : NULL;
  free(path);
  if (!wide) {
    return_refusal(c, "the path is empty, not UTF-8 or holds a zero byte");
    return;
  }
  OrtStatus* status = training->SaveCheckpoint(t->checkpoint, wide, jdl_arg_bool(c, 2) != 0);
  free(wide);
  if (status) {
    return_status(c, "cannot save the checkpoint", status);
  } else {
    return_ok(c);
  }
}

/**
 * Exports an inference model of a trainer's eval model and current parameters: arguments are
 * the trainer, the path, and the names of the outputs the inference model gives, each ending in
 * a zero byte. STATUS_OK, or STATUS_ERROR and why it could not be exported.
 */
static void trainer_export_inference(jdlo_ctx* c) {
  trainer* t = arg_trainer(c);
  if (!t) return;
  const uint8_t* names = jdl_arg_bytes_data(c, 2);
  uint32_t names_len = jdl_arg_bytes_len(c, 2);
  if (names_len == 0 || names[names_len - 1] != 0) {
    jdl_fail(c, "the output names are malformed");
    return;
  }
  size_t count = 0;
  for (uint32_t i = 0; i < names_len; i++) count += names[i] == 0;
  const char** list = malloc(count * sizeof *list);
  if (!list) {
    jdl_fail(c, "out of memory");
    return;
  }
  for (size_t i = 0, at = 0; i < count; i++) {
    list[i] = (const char*)names + at;
    at += strlen(list[i]) + 1;
  }
  char* path = arg_path(c, 1);
  ORTCHAR_T* wide = path ? ort_path(path) : NULL;
  free(path);
  if (!wide) {
    free(list);
    return_refusal(c, "the path is empty, not UTF-8 or holds a zero byte");
    return;
  }
  OrtStatus* status = training->ExportModelForInferencing(t->session, wide, count, list);
  free(wide);
  free(list);
  if (status) {
    return_status(c, "cannot export the inference model", status);
  } else {
    return_ok(c);
  }
}

static const jello_jdll_export EXPORTS[] = {
    {"training_available", training_available, JELLO_T_BOOL, 0, {0}, 0},
    {"trainer_open", trainer_open, JELLO_T_DYNAMIC, 5,
     {JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_I32},
     JELLO_JDLL_BLOCKING},
    {"trainer_open_bytes", trainer_open_bytes, JELLO_T_DYNAMIC, 5,
     {JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_BYTES, JELLO_T_I32},
     JELLO_JDLL_BLOCKING},
    {"trainer_close", trainer_close, JELLO_T_VOID, 1, {JELLO_T_DYNAMIC}, 0},
    {"trainer_train_step", trainer_train_step, JELLO_T_BYTES, 2, {JELLO_T_DYNAMIC, JELLO_T_BYTES},
     JELLO_JDLL_BLOCKING},
    {"trainer_eval_step", trainer_eval_step, JELLO_T_BYTES, 2, {JELLO_T_DYNAMIC, JELLO_T_BYTES},
     JELLO_JDLL_BLOCKING},
    {"trainer_save_checkpoint", trainer_save_checkpoint, JELLO_T_BYTES, 3,
     {JELLO_T_DYNAMIC, JELLO_T_BYTES, JELLO_T_BOOL}, JELLO_JDLL_BLOCKING},
    {"trainer_export_inference", trainer_export_inference, JELLO_T_BYTES, 3,
     {JELLO_T_DYNAMIC, JELLO_T_BYTES, JELLO_T_BYTES}, JELLO_JDLL_BLOCKING},
};

JELLO_JDLL_LIBRARY(oracle_train, EXPORTS);
