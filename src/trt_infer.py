import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Handle TensorRT inference flow natively on Jetson
class TrtInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load and deserialize engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Allocate memory buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, face_blob, body_blob):
        # Flatten and load inputs
        self.inputs[0]['host'][:] = face_blob.ravel()
        self.inputs[1]['host'][:] = body_blob.ravel()
        
        # Async transfer to device
        for i in self.inputs:
            cuda.memcpy_htod_async(i['device'], i['host'], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Async transfer back to host
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o['host'], o['device'], self.stream)
        
        self.stream.synchronize()
        return [o['host'] for o in self.outputs]