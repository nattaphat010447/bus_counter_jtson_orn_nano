import logging
import numpy as np

# TensorRT และ PyCUDA import แบบ lazy เพื่อไม่ crash บนเครื่องที่ไม่มี GPU
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # ไม่ import pycuda.autoinit ที่ระดับ module — init ใน __init__ แทน
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrtInference:
    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise ImportError("tensorrt และ pycuda จำเป็นสำหรับ TrtInference")

        # Initialize CUDA context ใน instance เพื่อไม่ผูกกับ module-level autoinit
        cuda.init()
        self._cuda_device  = cuda.Device(0)
        self._cuda_context = self._cuda_device.make_context()

        self.logger  = trt.Logger(trt.Logger.WARNING)
        self.stream  = cuda.Stream()
        self.inputs  = []
        self.outputs = []
        self.bindings = []

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        logger.info("TrtInference engine loaded: %s", engine_path)

        for binding in self.engine:
            size      = trt.volume(self.engine.get_binding_shape(binding))
            dtype     = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem  = cuda.pagelocked_empty(size, dtype)
            dev_mem   = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(dev_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': dev_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': dev_mem})

    def infer(self, face_blob: np.ndarray, body_blob: np.ndarray):
        self._cuda_context.push()
        try:
            self.inputs[0]['host'][:] = face_blob.ravel()
            self.inputs[1]['host'][:] = body_blob.ravel()

            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

            self.stream.synchronize()
            return [out['host'] for out in self.outputs]
        finally:
            self._cuda_context.pop()

    def close(self) -> None:
        """คืน CUDA resources — เรียกเมื่อใช้งานเสร็จ"""
        try:
            self._cuda_context.push()
            del self.inputs, self.outputs, self.bindings
            del self.context, self.engine, self.stream
        finally:
            self._cuda_context.pop()
            self._cuda_context.detach()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # ถ้า CUDA context หมดแล้วก็ข้ามไป

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()