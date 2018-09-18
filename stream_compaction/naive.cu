#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
		// Kernel for naive prefix scan
		__global__ void kernNaiveScan(int n, int *odata, const int *idata, const int offset) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;

			if (index >= offset) {
				odata[index] = idata[index - offset] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		void scan(int n, int *odata, const int *idata) {
			int *dev_idata, *dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			const int blockSize = 512;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			for (int d = 1; d <= ilog2ceil(n); ++d) {
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, pow(2, d - 1));
				std::swap(dev_odata, dev_idata);
			}

            timer().endGpuTimer();

			cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
