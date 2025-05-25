#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>

extern "C"
{
   int run_knn_distance(const float *train_data, const float *test_point, float *output_distances, int train_len, int dims)
   {
      @autoreleasepool
      {
         id<MTLDevice> device = MTLCreateSystemDefaultDevice();
         id<MTLCommandQueue> queue = [device newCommandQueue];

         NSError *error = nil;
         NSString *src = [NSString stringWithContentsOfFile:@"knn_kernel.metal"
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
         if (error || !src)
         {
            return -1;
         }

         id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
         if (!lib)
         {
            return -2;
         }

         id<MTLFunction> func = [lib newFunctionWithName:@"knn_distance"];
         if (!func)
         {
            return -3;
         }

         id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:nil];
         if (!pipeline)
         {
            return -4;
         }

         NSUInteger train_data_size = sizeof(float) * train_len * dims;
         NSUInteger test_point_size = sizeof(float) * dims;
         NSUInteger output_size = sizeof(float) * train_len;

         id<MTLBuffer> trainBuf = [device newBufferWithBytes:train_data length:train_data_size options:0];
         id<MTLBuffer> testBuf = [device newBufferWithBytes:test_point length:test_point_size options:0];
         id<MTLBuffer> outputBuf = [device newBufferWithLength:output_size options:MTLResourceStorageModeShared];

         id<MTLCommandBuffer> cmd = [queue commandBuffer];
         id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

         [encoder setComputePipelineState:pipeline];
         [encoder setBuffer:trainBuf offset:0 atIndex:0];
         [encoder setBuffer:testBuf offset:0 atIndex:1];
         [encoder setBuffer:outputBuf offset:0 atIndex:2];

         MTLSize grid = MTLSizeMake(train_len, 1, 1);
         MTLSize threads = MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup, 1, 1);
         [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
         [encoder endEncoding];

         [cmd commit];
         [cmd waitUntilCompleted];

         memcpy((void *)output_distances, outputBuf.contents, output_size);
         return 0;
      }
   }

   const char *metal_get_gpu_name()
   {
      static std::string gpuName;

      @autoreleasepool
      {
         id<MTLDevice> device = MTLCreateSystemDefaultDevice();
         if (device)
         {
            gpuName = std::string([[device name] UTF8String]);
         }
         else
         {
            gpuName = "No Metal-compatible GPU found";
         }
      }

      return gpuName.c_str();
   }
}
