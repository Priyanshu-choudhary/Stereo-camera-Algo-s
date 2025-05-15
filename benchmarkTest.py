import cv2
import numpy as np
import time
import os
from collections import defaultdict

CALIB_DATA_DIR = "./calibration_data"
LEFT_IMG_PATH = "./calibrationImages/VGAimages/imageLeft50.png"
RIGHT_IMG_PATH = "./calibrationImages/VGAimages/imageRight50.png"

class RectificationBenchmark:
    def __init__(self, left_img, right_img, left_map, right_map, runs=100):
        self.left_img = left_img
        self.right_img = right_img
        self.left_map_x = left_map[0].astype(np.float32)
        self.left_map_y = left_map[1].astype(np.float32)
        self.right_map_x = right_map[0].astype(np.float32)
        self.right_map_y = right_map[1].astype(np.float32)
        self.runs = runs
        self.results = defaultdict(list)
        
        # Warm up CUDA
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            dummy = cv2.cuda_GpuMat()
            dummy.upload(np.zeros((10,10), np.uint8))
    
    def benchmark_cpu(self):
        print("=== CPU Benchmark ===")
        # Test different interpolation methods
        for interp in [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]:
            start = time.perf_counter()
            for _ in range(self.runs):
                rect_left = cv2.remap(self.left_img, 
                                    self.left_map_x, 
                                    self.left_map_y, 
                                    interp)
                rect_right = cv2.remap(self.right_img, 
                                      self.right_map_x, 
                                      self.right_map_y, 
                                      interp)
            elapsed = (time.perf_counter() - start) / self.runs * 1000
            self.results['CPU'].append((f"INTER_{interp}", elapsed))
            print(f"CPU {interp}: {elapsed:.2f}ms")
    
    def benchmark_gpu_naive(self):
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return
            
        print("\n=== GPU Naive Benchmark ===")
        # Upload images each time (like your current approach)
        for interp in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
            start = time.perf_counter()
            for _ in range(self.runs):
                gpu_left = cv2.cuda_GpuMat()
                gpu_right = cv2.cuda_GpuMat()
                gpu_left.upload(self.left_img)
                gpu_right.upload(self.right_img)
                
                gpu_map1_left = cv2.cuda_GpuMat()
                gpu_map2_left = cv2.cuda_GpuMat()
                gpu_map1_right = cv2.cuda_GpuMat()
                gpu_map2_right = cv2.cuda_GpuMat()
                gpu_map1_left.upload(self.left_map_x)
                gpu_map2_left.upload(self.left_map_y)
                gpu_map1_right.upload(self.right_map_x)
                gpu_map2_right.upload(self.right_map_y)
                
                rect_left = cv2.cuda.remap(gpu_left, 
                                          gpu_map1_left, 
                                          gpu_map2_left, 
                                          interp).download()
                rect_right = cv2.cuda.remap(gpu_right,
                                           gpu_map1_right,
                                           gpu_map2_right,
                                           interp).download()
            elapsed = (time.perf_counter() - start) / self.runs * 1000
            self.results['GPU_Naive'].append((f"INTER_{interp}", elapsed))
            print(f"GPU Naive {interp}: {elapsed:.2f}ms")
    
    def benchmark_gpu_optimized(self):
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return
            
        print("\n=== GPU Optimized Benchmark ===")
        # Pre-upload maps (like your current implementation)
        gpu_map1_left = cv2.cuda_GpuMat()
        gpu_map2_left = cv2.cuda_GpuMat()
        gpu_map1_right = cv2.cuda_GpuMat()
        gpu_map2_right = cv2.cuda_GpuMat()
        gpu_map1_left.upload(self.left_map_x)
        gpu_map2_left.upload(self.left_map_y)
        gpu_map1_right.upload(self.right_map_x)
        gpu_map2_right.upload(self.right_map_y)
        
        for interp in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
            start = time.perf_counter()
            for _ in range(self.runs):
                gpu_left = cv2.cuda_GpuMat()
                gpu_right = cv2.cuda_GpuMat()
                gpu_left.upload(self.left_img)
                gpu_right.upload(self.right_img)
                
                rect_left = cv2.cuda.remap(gpu_left, 
                                          gpu_map1_left, 
                                          gpu_map2_left, 
                                          interp).download()
                rect_right = cv2.cuda.remap(gpu_right,
                                           gpu_map1_right,
                                           gpu_map2_right,
                                           interp).download()
            elapsed = (time.perf_counter() - start) / self.runs * 1000
            self.results['GPU_Opt'].append((f"INTER_{interp}", elapsed))
            print(f"GPU Opt {interp}: {elapsed:.2f}ms")
    
    def benchmark_gpu_streamed(self):
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return
            
        print("\n=== GPU Streamed Benchmark ===")
        # Use CUDA streams for async operations
        stream = cv2.cuda_Stream()
        
        # Pre-upload maps
        gpu_map1_left = cv2.cuda_GpuMat()
        gpu_map2_left = cv2.cuda_GpuMat()
        gpu_map1_right = cv2.cuda_GpuMat()
        gpu_map2_right = cv2.cuda_GpuMat()
        gpu_map1_left.upload(self.left_map_x, stream=stream)
        gpu_map2_left.upload(self.left_map_y, stream=stream)
        gpu_map1_right.upload(self.right_map_x, stream=stream)
        gpu_map2_right.upload(self.right_map_y, stream=stream)
        
        for interp in [cv2.INTER_LINEAR]:
            start = time.perf_counter()
            for _ in range(self.runs):
                gpu_left = cv2.cuda_GpuMat()
                gpu_right = cv2.cuda_GpuMat()
                gpu_left.upload(self.left_img, stream=stream)
                gpu_right.upload(self.right_img, stream=stream)
                
                rect_left = cv2.cuda.remap(gpu_left, 
                                         gpu_map1_left, 
                                         gpu_map2_left, 
                                         interp,
                                         stream=stream)
                rect_right = cv2.cuda.remap(gpu_right,
                                          gpu_map1_right,
                                          gpu_map2_right,
                                          interp,
                                          stream=stream)
                
                # Only sync at the end if we need the results
                stream.waitForCompletion()
                
                # For benchmarking, we don't actually need to download
                # rect_left.download(stream=stream)
                # rect_right.download(stream=stream)
            elapsed = (time.perf_counter() - start) / self.runs * 1000
            self.results['GPU_Stream'].append((f"INTER_{interp}", elapsed))
            print(f"GPU Stream {interp}: {elapsed:.2f}ms")
    
    def benchmark_all(self):
        self.benchmark_cpu()
        self.benchmark_gpu_naive()
        self.benchmark_gpu_optimized()
        self.benchmark_gpu_streamed()
        return self.results

# Usage example:
if __name__ == "__main__":
    # Load your calibration data and images
     # Load separate x/y maps
    left_map_x = np.load(os.path.join(CALIB_DATA_DIR, "left_map_x.npy"))
    left_map_y = np.load(os.path.join(CALIB_DATA_DIR, "left_map_y.npy"))
    right_map_x = np.load(os.path.join(CALIB_DATA_DIR, "right_map_x.npy"))
    right_map_y = np.load(os.path.join(CALIB_DATA_DIR, "right_map_y.npy"))
    
    # Combine into OpenCV-friendly format
    left_map = (left_map_x, left_map_y)
    right_map = (right_map_x, right_map_y)
    
    
                             
    left_img = cv2.imread(LEFT_IMG_PATH)
    right_img = cv2.imread(RIGHT_IMG_PATH)
    
    # Run benchmarks
    benchmark = RectificationBenchmark(left_img, right_img, left_map, right_map, runs=50)
    results = benchmark.benchmark_all()
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for method, timings in results.items():
        for interp, time_ms in timings:
            print(f"{method:12s} {interp:15s}: {time_ms:.2f}ms per frame")