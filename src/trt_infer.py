import logging
import numpy as np

# TensorRT และ PyCUDA import แบบ lazy เพื่อไม่ crash บนเครื่องที่ไม่มี GPU
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrtInference:
    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise ImportError("tensorrt และ pycuda จำเป็นสำหรับ TrtInference")

        # Initialize CUDA context
        cuda.init()
        self._cuda_device  = cuda.Device(0)
        self._cuda_context = self._cuda_device.make_context()

        self.logger  = trt.Logger(trt.Logger.WARNING)
        self.stream  = cuda.Stream()
        self.inputs  = []
        self.outputs = []

        # Load Engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        logger.info("TrtInference engine loaded (TRT10 API): %s", engine_path)

        # TRT10 API: Iterating over tensor names instead of binding indices
        for tensor_name in self.engine:
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size  = trt.volume(shape)
            
            # ป้องกันกรณี shape มีปัญหา ให้เป็นค่าบวกเสมอ
            if size < 0: size = abs(size)

            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem  = cuda.mem_alloc(host_mem.nbytes)

            # ผูก Address ของ Memory เข้ากับชื่อ Tensor
            self.context.set_tensor_address(tensor_name, int(dev_mem))

            # เช็คว่าเป็น Input หรือ Output
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': dev_mem, 'name': tensor_name})
            else:
                self.outputs.append({'host': host_mem, 'device': dev_mem, 'name': tensor_name})

    def infer(self, face_blob: np.ndarray, body_blob: np.ndarray):
        self._cuda_context.push()
        try:
            # Load inputs (ดึงข้อมูลเข้า host memory)
            if len(self.inputs) > 0:
                self.inputs[0]['host'][:] = face_blob.ravel()
            if len(self.inputs) > 1:
                self.inputs[1]['host'][:] = body_blob.ravel()

            # Async transfer Host -> Device
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

            # Execute TRT10 Async (ใช้ v3 แทน v2)
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Async transfer Device -> Host
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
            del self.inputs, self.outputs
            del self.context, self.engine, self.stream
        finally:
            self._cuda_context.pop()
            self._cuda_context.detach()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass 

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()