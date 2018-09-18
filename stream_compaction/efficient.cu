#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// Kernels for efficient prefix scan
		__global__ void kernUpScan(int n, int *data, const int offset, const int offset2) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;

			if (index % offset2 != 0)
				return;

			data[index + offset2 - 1] += data[index + offset - 1];
		}

		__global__ void kernDownScan(int n, int *data, const int offset, const int offset2) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n)
				return;

			if (index % offset2 != 0)
				return;

			int temp = data[index + offset - 1];
			data[index + offset - 1] = data[index + offset2 - 1];
			data[index + offset2 - 1] += temp;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		void scanNoTimer(int n, int *odata, const int *idata) {
			int *dev_data;
			cudaMalloc((void**)&dev_data, n * sizeof(int));
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			const int blockSize = 256;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			for (int d = 0; d < ilog2ceil(n); ++d) {
				int offset = pow(2, d);
				kernUpScan << <fullBlocksPerGrid, blockSize >> > (n, dev_data, offset, offset * 2);
			}

			cudaMemset(dev_data + n - 1, 0, 1);
			for (int d = ilog2ceil(n); d >= 0; --d) {
				int offset = pow(2, d);
				kernDownScan << <fullBlocksPerGrid, blockSize >> > (n, dev_data, offset, offset * 2);
			}

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_data);
		}

        void scan(int n, int *odata, const int *idata) {
			timer().startGpuTimer();
			scanNoTimer(n, odata, idata);
			timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			int *dev_idata, *dev_odata, *dev_mapped, *dev_scanned;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_mapped, n * sizeof(int));
			cudaMalloc((void**)&dev_scanned, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			
			const int blockSize = 256;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
			Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_mapped, dev_idata);
			scanNoTimer(n, dev_scanned, dev_mapped);
			Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_mapped, dev_scanned);
            timer().endGpuTimer();

			int count, lastbool;
			cudaMemcpy(&count, dev_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lastbool, dev_mapped + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			count += lastbool;
			
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_mapped);
			cudaFree(dev_scanned);

            return count;
        }
    }
}
