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
		__global__ void kernNaiveScan(int n, int *odata, const int *idata, int ilog2) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= n)
				return;

			for (int d = 1; d < n; d *= 2) {
				if (k >= d) {
					int idatakd = idata[k - d];
					int idatak = idata[k];
					odata[k] = idatakd + idatak;
				}
				else {
					odata[k] = idata[k];
				}
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int blockSize = ilog2ceil(n);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			kernNaiveScan << <fullBlocksPerGrid, blockSize >> >(n, odata, idata, ilog2ceil(n));
            timer().endGpuTimer();
        }
    }
}
