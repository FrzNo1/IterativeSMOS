#include <stdio.h>
#include <stdlib.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <limits>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//#define SAFE


namespace IterativeSMOS {
    using namespace std;



#define MAX_THREADS_PER_BLOCK 1024
#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)




    /// ***********************************************************
    /// ***********************************************************
    /// **** Safety FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************




    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER CPU FUNCTIONS
    /// ***********************************************************
    /// ***********************************************************

    /* This function initializes a vector to all zeros on the host (CPU).
     */
    template<typename T>
    void setToAllZero (T * d_vector, int length) {
        cudaMemset(d_vector, 0, length * sizeof(T));
    }


    /* This function finds the bin containing the kth element we are looking for (works on
       the host). While doing the scan, it stores the sum-so-far of the number of elements in
       the buckets where k values fall into.

       markedBuckets : buckets containing the corresponding k values
       sums : sum-so-far of the number of elements in the buckets where k values fall into
    */
    inline int findKBuckets(unsigned int * d_bucketCount, unsigned int * h_bucketCount, int numBuckets
            , unsigned int * kVals, int numKs, unsigned int * sums, unsigned int * markedBuckets
            , int numBlocks) {
        // consider the last row which holds the total counts
        int sumsRowIndex= numBuckets * (numBlocks-1);

        cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex,
                   sizeof(unsigned int) * numBuckets, cudaMemcpyDeviceToHost);

        int kBucket = 0;
        int k;
        int sum = h_bucketCount[0];

        for(register int i = 0; i < numKs; i++) {
            k = kVals[i];
            while ((sum < k) & (kBucket < numBuckets - 1)) {
                kBucket++;
                sum += h_bucketCount[kBucket];
            }
            markedBuckets[i] = kBucket;
            sums[i] = sum - h_bucketCount[kBucket];
        }

        return 0;
    }

    template <typename T>
    inline int updatekVals_iterative(unsigned int * kVals, int * numKs, T * output, unsigned int * kIndicies,
                                     int * length, int * lengthOld, unsigned int * h_bucketCount, unsigned int * markedBuckets,
                                     unsigned int * kthBucketScanner, unsigned int * reindexCounter,
                                     unsigned int * uniqueBuckets, unsigned int * uniqueBucketCounts,
                                     int * numUniqueBuckets, int * numUniqueBucketsOld,
                                     unsigned int * tempKorderBucket, unsigned int * tempKorderIndeces, int * tempKorderLength) {
        int index = 0;
        int numKsindex = 0;
        *numUniqueBucketsOld = *numUniqueBuckets;
        *numUniqueBuckets = 0;
        *lengthOld = *length;
        *tempKorderLength = 0;

        while (index < *numKs) {
            if (h_bucketCount[markedBuckets[index]] == 1) {
                tempKorderIndeces[*tempKorderLength] = kIndicies[index];
                tempKorderBucket[*tempKorderLength] = markedBuckets[index];
                (*tempKorderLength)++;
                index++;
                continue;
            }

            break;
        }

        if (index < *numKs) {
            uniqueBuckets[0] = markedBuckets[index];
            uniqueBucketCounts[0] = h_bucketCount[markedBuckets[index]];
            reindexCounter[0] = 0;
            *numUniqueBuckets = 1;
            kVals[0] = kVals[index] - kthBucketScanner[index];
            kIndicies[0] = kIndicies[index];
            numKsindex++;
            index++;
        }

        for ( ; index < *numKs; index++) {

            if (h_bucketCount[markedBuckets[index]] == 1) {
                tempKorderIndeces[*tempKorderLength] = kIndicies[index];
                tempKorderBucket[*tempKorderLength] = markedBuckets[index];
                (*tempKorderLength)++;
                continue;
            }

            if (markedBuckets[index] != uniqueBuckets[(*numUniqueBuckets) - 1]) {
                uniqueBuckets[*numUniqueBuckets] = markedBuckets[index];
                uniqueBucketCounts[*numUniqueBuckets] = h_bucketCount[markedBuckets[index]];
                reindexCounter[*numUniqueBuckets] = reindexCounter[(*numUniqueBuckets) - 1]
                                                    + h_bucketCount[markedBuckets[numKsindex - 1]];
                (*numUniqueBuckets)++;
            }
            kVals[numKsindex] = reindexCounter[(*numUniqueBuckets) - 1] + kVals[index] - kthBucketScanner[index];
            kIndicies[numKsindex] = kIndicies[index];
            numKsindex++;
        }

        *numKs = numKsindex;

        if (*numKs > 0)
            *length = reindexCounter[(*numUniqueBuckets) - 1] + h_bucketCount[markedBuckets[numKsindex - 1]];


        return 0;
    }

    template <typename T>	
    void swapPointers(T** a, T** b) {
        T * temp = * a;
        * a = * b;
        * b = temp;
    }



    /// ***********************************************************
    /// ***********************************************************
    /// **** HELPER GPU FUNCTIONS-KERNELS
    /// ***********************************************************
    /// ***********************************************************


    /*
     *
     */
    template <typename T>
    __global__ void generateBucketsandSlopes_iterative (T * pivotsLeft, T * pivotsRight, double * slopes,
                                                        unsigned int * uniqueBucketsCounts, int numUniqueBuckets,
                                                        unsigned int * kthnumBuckets, int length, int offset, int numBuckets) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // Assign bucket number and slope to first to the second to last active buckets
        if (index < numUniqueBuckets - 1) {
            for (int i = index; i < numUniqueBuckets - 1; i += offset) {

                // assign bucket number
                kthnumBuckets[i] = max(uniqueBucketsCounts[i] * numBuckets / length ,2);

                // assign slope
                slopes[i] = (double) kthnumBuckets[i] / (double) (pivotsRight[i] - pivotsLeft[i]);  // potential problems, last number
                // will not go into the bucket
            }
        }

        __syncthreads();



        // Assign bucket number and slope to the last active buckets
        if (index < 1) {
            // exclusive cumulative sum to the kthnumbuckets for finding the correct number of buckets for the last active buckets
            // thrust::exclusive_scan(thrust::host, kthnumBuckets, kthnumBuckets + numUniqueBuckets, kthnumBuckets, 0);

            // my own version of exclusive scan
	    if (numUniqueBuckets > 1) {
		for (int i = 1; i < numUniqueBuckets - 1; i++) {
		    kthnumBuckets[i] += kthnumBuckets[i - 1];
		}

		for (int i = numUniqueBuckets - 1; i > 0; i--) {
		    kthnumBuckets[i] = kthnumBuckets[i - 1];
		}
	    }

	    kthnumBuckets[0] = 0;
            // bucket number is assigned automatically

            // assign slope
            slopes[numUniqueBuckets - 1] = (numBuckets - kthnumBuckets[numUniqueBuckets - 1])
                                           / (double) (pivotsRight[numUniqueBuckets - 1] - pivotsLeft[numUniqueBuckets - 1]);
        }
    }


    /* This function assigns elements to buckets based on the pivots and slopes determined
       by a randomized sampling of the elements in the vector. At the same time, this
       function keeps track of count.

       d_elementToBucket : bucket assignment for every array element
       d_bucketCount : number of element that falls into the indexed buckets within the block
    */
    template <typename T>
    __global__ void assignSmartBucket_iterative(T * d_vector, int length, unsigned int * d_elementToBucket,
                                                double * slopes, T * pivotsLeft, T * pivotsRight,
                                                unsigned int * kthNumBuckets, unsigned int * d_bucketCount,
                                                int numUniqueBuckets, int numBuckets, int offset) {

        int index = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int bucketIndex;
        int threadIndex = threadIdx.x;

        //variables in shared memory for fast access
        extern __shared__ unsigned int array[];
        double * sharedSlopes = (double *)array;
        T * sharedPivotsLeft = (T *)&sharedSlopes[numUniqueBuckets];
        unsigned int * sharedkthNumBuckets = (unsigned int *)&sharedPivotsLeft[numUniqueBuckets];
        unsigned int * sharedBuckets = (unsigned int *)&sharedkthNumBuckets[numUniqueBuckets];

        //reading bucket counts into shared memory where increments will be performed
        for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) {

            if (threadIndex < numBuckets)
                sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;
        }


        if (threadIndex < numUniqueBuckets) {
            sharedPivotsLeft[threadIndex] = pivotsLeft[threadIndex];
            sharedSlopes[threadIndex] = slopes[threadIndex];
            sharedkthNumBuckets[threadIndex] = kthNumBuckets[threadIndex];
        }

        __syncthreads();

        //assigning elements to buckets and incrementing the bucket counts
        if (index < length) {

            for (int i = index; i < length; i += offset) {
                T num = d_vector[i];
                int minPivotIndex = 0;
                int maxPivotIndex = numUniqueBuckets - 1;
                int midPivotIndex;

                // find the index of left pivots that is greatest s.t. lower than or equal to
                // num using binary search
                if (num >= pivotsLeft[numUniqueBuckets - 1]) {
                    minPivotIndex = numUniqueBuckets - 1;
                }
                else {
                    for (int j = 1; j < numUniqueBuckets - 1; j *= 2) {
                        midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
                        if (num >= pivotsLeft[midPivotIndex])
                            minPivotIndex = midPivotIndex;
                        else
                            maxPivotIndex = midPivotIndex;
                    }
                }

                bucketIndex = sharedkthNumBuckets[minPivotIndex]
                              + (int) (((double)num - (double)sharedPivotsLeft[minPivotIndex])
                                       * sharedSlopes[minPivotIndex]);

                for (int j = 0; j < numUniqueBuckets; j++) {
                    if (num == pivotsRight[j]) {
                        bucketIndex--;
                        break;
                    }

                }

                d_elementToBucket[i] = bucketIndex;
                atomicInc(sharedBuckets + bucketIndex, length);
            }
        }

        __syncthreads();

        //reading bucket counts from shared memory back to global memory
        for (int i = 0; i <(numBuckets / MAX_THREADS_PER_BLOCK); i++)
            if (threadIndex < numBuckets)
                *(d_bucketCount + blockIdx.x * numBuckets
                  + i * MAX_THREADS_PER_BLOCK + threadIndex) =
                        *(sharedBuckets + i * MAX_THREADS_PER_BLOCK + threadIndex);
    }

    /* This function cumulatively sums the count of every block for a given bucket s.t. the
       last block index holds the total number of elements falling into that bucket all over the
       array.
       updates d_bucketCount
    */
    __global__ void sumCounts(unsigned int * d_bucketCount, const int numBuckets
            , const int numBlocks) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        for(int j=1; j<numBlocks; j++)
            d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    }


    /* This function reindexes the buckets counts for every block according to the
       accumulated d_reindexCounter counter for the reduced vector.
       updates d_bucketCount
    */
    __global__ void reindexCounts(unsigned int * d_bucketCount, int numBuckets, int numBlocks,
                                  unsigned int * d_reindexCounter, unsigned int * d_uniqueBuckets,
                                  const int numUniqueBuckets) {
        int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadIndex < numUniqueBuckets) {
            int index = d_uniqueBuckets[threadIndex];
            unsigned int add = d_reindexCounter[threadIndex];

            for (int j = 0; j < numBlocks; j++)
                d_bucketCount[index + numBuckets * j] += add;
        }
    }

    /* This function copies the elements of buckets that contain kVals into a newly allocated
       reduced vector space.
       newArray - reduced size vector containing the essential elements
    */
    template <typename T>
    __global__ void copyElements_iterative (T * d_vector, T * d_newvector, int lengthOld, unsigned int * elementToBuckets,
                                            unsigned int * uniqueBuckets, int numUniqueBuckets,
                                            unsigned int * d_bucketCount, int numBuckets, unsigned int offset) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int threadIndex;
        int loop = numBuckets / MAX_THREADS_PER_BLOCK;

        extern __shared__ unsigned int sharedBuckets[];

        for (int i = 0; i <= loop; i++) {
            threadIndex = i * blockDim.x + threadIdx.x;
            if (threadIndex < numUniqueBuckets)
                sharedBuckets[threadIndex] = uniqueBuckets[threadIndex];
        }

        __syncthreads();


        if (index < lengthOld) {

            for (int i = index; i < lengthOld; i += offset) {
                unsigned int temp = elementToBuckets[i];
                int minBucketIndex = 0;
                int maxBucketIndex = numUniqueBuckets - 1;
                int midBucketIndex;

                for (int j = 1; j < numUniqueBuckets; j *= 2) {
                    midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
                    if (temp > sharedBuckets[midBucketIndex])
                        minBucketIndex = midBucketIndex + 1;
                    else
                        maxBucketIndex = midBucketIndex;
                }

                if (temp == sharedBuckets[maxBucketIndex])
                    d_newvector[atomicDec(d_bucketCount + blockIdx.x * numBuckets
                                          + sharedBuckets[maxBucketIndex], lengthOld) - 1] = d_vector[i];
            }
        }

        // needs to swap d_vector with d_newvector
    }


    /* This function copies the elements of buckets that contain kVals into a newly allocated
       reduced vector space.
       newArray - reduced size vector containing the essential elements
    */
    template <typename T>
    __global__ void updatePivots_iterative(T * d_pivotsLeft, T * d_newPivotsLeft, T * d_newPivotsRight,
                                           double * slopes, unsigned int * kthnumBuckets, unsigned int * uniqueBuckets,
                                           int numUniqueBuckets, int numUniqueBucketsOld, int offset) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < numUniqueBuckets) {
            for (int i = index; i < numUniqueBuckets; i += offset) {

                // perform binary search to find kthNumBucket that is greatest s.t. lower than or equal to the bucket
                unsigned int bucket = uniqueBuckets[i];
                int minBucketIndex = 0;
                int maxBucketIndex = numUniqueBucketsOld - 1;
                int midBucketIndex;

                if (bucket >= kthnumBuckets[numUniqueBucketsOld - 1]) {
                    minBucketIndex = numUniqueBucketsOld - 1;
                }
                else {
                    for (int j = 1; j < numUniqueBucketsOld - 1; j *= 2) {
                        midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
                        if (bucket >= kthnumBuckets[midBucketIndex])
                            minBucketIndex = midBucketIndex;
                        else
                            maxBucketIndex = midBucketIndex;
                    }
                }

                d_newPivotsLeft[i] = d_pivotsLeft[minBucketIndex] +
                                     (T)(((double)(bucket - kthnumBuckets[minBucketIndex])) / slopes[minBucketIndex]);
                d_newPivotsRight[i] = d_pivotsLeft[minBucketIndex] +
                                      (T)(((double)(bucket - kthnumBuckets[minBucketIndex] + 1)) / slopes[minBucketIndex]);
            }
        }

        // needs to swap pointers of pivotsLeft with newPivotsLeft, pivotsRight with newPivotsRight
    }


    template <typename T>
    __global__ void updateOutput_iterative (T * d_vector, unsigned int * d_elementToBucket, int lengthOld, T * d_tempOutput,
                                            unsigned int * d_tempKorderBucket, int tempKorderLength, int offset){

        int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index < lengthOld) {
            for (int i = index; i < lengthOld; i += offset) {
                unsigned int bucket = d_elementToBucket[i];

                for (int j = 0; j < tempKorderLength; j++) {
                    if (d_tempKorderBucket[j] == bucket)
                        d_tempOutput[j] = d_vector[i];
                }
            }
        }
    }




    /// ***********************************************************
    /// ***********************************************************
    /// **** GENERATE KD PIVOTS
    /// ***********************************************************
    /// ***********************************************************

    /* Hash function using Monte Carlo method
     */
    __host__ __device__
    unsigned int hash(unsigned int a) {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }



    /* RandomNumberFunctor
     */
    struct RandomNumberFunctor :
            public thrust::unary_function<unsigned int, float> {
        unsigned int mainSeed;

        RandomNumberFunctor(unsigned int _mainSeed) :
                mainSeed(_mainSeed) {}

        __host__ __device__
        float operator()(unsigned int threadIdx)
        {
            unsigned int seed = hash(threadIdx) * mainSeed;

            thrust::default_random_engine rng(seed);
            rng.discard(threadIdx);
            thrust::uniform_real_distribution<float> u(0, 1);

            return u(rng);
        }
    };



    /* This function creates a random vector of 1024 elements in the range [0 1]
     */
    template <typename T>
    void createRandomVector(T * d_vec, int size) {
        timeval t1;
        unsigned int seed;

        gettimeofday(&t1, NULL);
        seed = t1.tv_usec * t1.tv_sec;
        // seed = 10;

        thrust::device_ptr<T> d_ptr(d_vec);
        thrust::transform (thrust::counting_iterator<unsigned int>(0),
                           thrust::counting_iterator<unsigned int>(size),
                           d_ptr, RandomNumberFunctor(seed));
    }



    /* This function maps the [0 1] range to the [0 vectorSize] and
       grabs the corresponding elements.
    */
    template <typename T>
    __global__ void enlargeIndexAndGetElements (T * in, T * list, int size) {
        *(in + blockIdx.x*blockDim.x + threadIdx.x) =
                *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
    }

    __global__ void enlargeIndexAndGetElements (float * in, unsigned int * out, unsigned int * list, int size) {
        *(out + blockIdx.x * blockDim.x + threadIdx.x) =
                (unsigned int) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
    }



    /* This function generates Pivots from the random sampled data and calculates slopes.

       pivots - arrays of pivots
       slopes - array of slopes
    */
    template <typename T>
    void generatePivots (unsigned int * pivots, double * slopes, unsigned int * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, unsigned int min, unsigned int max) {

        float * d_randomFloats;
        unsigned int * d_randomInts;
        int endOffset = 22;
        int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
        int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

        cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);

        d_randomInts = (unsigned int *) d_randomFloats;

        createRandomVector (d_randomFloats, sizeOfSample);

        // converts randoms floats into elements from necessary indices
        enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
        , MAX_THREADS_PER_BLOCK>>>(d_randomFloats, d_randomInts, d_list,
                                   sizeOfVector);

        pivots[0] = min;
        pivots[numPivots-1] = max;

        thrust::device_ptr<T>randoms_ptr(d_randomInts);
        thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

        cudaThreadSynchronize();

        // set the pivots which are next to the min and max pivots using the random element
        // endOffset away from the ends
        cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (unsigned int)
                , cudaMemcpyDeviceToHost);
        cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1,
                    sizeof (unsigned int), cudaMemcpyDeviceToHost);
        slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

        for (register int i = 2; i < numPivots - 2; i++) {
            cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1,
                        sizeof (unsigned int), cudaMemcpyDeviceToHost);
            slopes[i - 1] = numSmallBuckets / (double) (pivots[i] - pivots[i - 1]);
        }

        slopes[numPivots - 3] = numSmallBuckets /
                                (double) (pivots[numPivots - 2] - pivots[numPivots - 3]);
        slopes[numPivots - 2] = numSmallBuckets /
                                (double) (pivots[numPivots - 1] - pivots[numPivots - 2]);

        cudaFree(d_randomFloats);
    }

    template <typename T>
    void generatePivots (T * pivots, double * slopes, T * d_list, int sizeOfVector
            , int numPivots, int sizeOfSample, int totalSmallBuckets, T min, T max) {
        T * d_randoms;
        int endOffset = 22;
        int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
        int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

        cudaMalloc (&d_randoms, sizeof (T) * sizeOfSample);

        createRandomVector (d_randoms, sizeOfSample);

        // converts randoms floats into elements from necessary indices
        enlargeIndexAndGetElements<<<(sizeOfSample/MAX_THREADS_PER_BLOCK)
        , MAX_THREADS_PER_BLOCK>>>(d_randoms, d_list, sizeOfVector);

        pivots[0] = min;
        pivots[numPivots - 1] = max;

        thrust::device_ptr<T>randoms_ptr(d_randoms);
        thrust::sort(randoms_ptr, randoms_ptr + sizeOfSample);

        cudaThreadSynchronize();

        // set the pivots which are endOffset away from the min and max pivots
        cudaMemcpy (pivots + 1, d_randoms + endOffset - 1, sizeof (T),
                    cudaMemcpyDeviceToHost);
        cudaMemcpy (pivots + numPivots - 2, d_randoms + sizeOfSample - endOffset - 1,
                    sizeof (T), cudaMemcpyDeviceToHost);
        slopes[0] = numSmallBuckets / ((double)pivots[1] - (double)pivots[0]);

        for (register int i = 2; i < numPivots - 2; i++) {
            cudaMemcpy (pivots + i, d_randoms + pivotOffset * (i - 1) + endOffset - 1,
                        sizeof (T), cudaMemcpyDeviceToHost);
            slopes[i - 1] = numSmallBuckets / ((double) pivots[i] - (double) pivots[i - 1]);
        }

        slopes[numPivots - 3] = numSmallBuckets /
                                ((double)pivots[numPivots - 2] - (double)pivots[numPivots - 3]);
        slopes[numPivots - 2] = numSmallBuckets /
                                ((double)pivots[numPivots - 1] - (double)pivots[numPivots - 2]);

        cudaFree(d_randoms);
    }




    /// ***********************************************************
    /// ***********************************************************
    /// **** iterativeSMOS: the main algorithm
    /// ***********************************************************
    /// ***********************************************************


    /* This function is the main process of the algorithm. It reduces the given multi-selection
       problem to a smaller problem by using bucketing ideas.
    */
    template <typename T>
    T iterativeSMOS (T* d_vector, int length, unsigned int * kVals, int numKs, T * output, int blocks
            , int threads, int numBuckets, int numPivots) {

        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.1: Find Min and Max of the whole vector
        /// **** We don't need to go through the rest of the algorithm if it's flat
        /// ***********************************************************

        //find max and min with thrust
        T maximum, minimum;

        thrust::device_ptr<T>dev_ptr(d_vector);
        thrust::pair<thrust::device_ptr<T>, thrust::device_ptr<T> > result =
                thrust::minmax_element(dev_ptr, dev_ptr + length);

        
        minimum = *result.first;
        maximum = *result.second;

        //if the max and the min are the same, then we are done
        if (maximum == minimum) {
            for (register int i = 0; i < numKs; i++)
                output[i] = minimum;

            return 1;
        }


        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.2: Declare variables and allocate memory
        /// **** Declare Variables
        /// ***********************************************************

        // declare variables for kernel launches
        int threadsPerBlock = threads;
        int numBlocks = blocks;
        int offset = blocks * threads;

        // variables for the randomized selection
        int sampleSize = 1024;

        // pivots variables
        int numMemory = max(numKs, numPivots);

        double * slopes = (double*)malloc(numMemory * sizeof(double));                  // size will be different
        double * d_slopes;
        T * pivots = (T*)malloc(numPivots * sizeof(T));
        T * d_pivots;
        CUDA_CALL(cudaMalloc(&d_slopes, numMemory * sizeof(double)));
        CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

        T * pivotsLeft = (T*)malloc(numMemory * sizeof(T));                                 // new variables
        T * pivotsRight = (T*)malloc(numMemory * sizeof(T));
        T * d_pivotsLeft;
        T * d_pivotsRight;
        T * newPivotsLeft = (T*)malloc(numMemory * sizeof(T));                            // potential not being used
        T * newPivotsRight = (T*)malloc(numMemory * sizeof(T));                           // potential not being used
        T * d_newPivotsLeft;
        T * d_newPivotsRight;
        CUDA_CALL(cudaMalloc(&d_pivotsLeft, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_pivotsRight, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_newPivotsLeft, numMemory * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_newPivotsRight, numMemory * sizeof(T)));


        //Allocate memory to store bucket assignments
        size_t size = length * sizeof(unsigned int);
        unsigned int * d_elementToBucket;    //array showing what bucket every element is in
        CUDA_CALL(cudaMalloc(&d_elementToBucket, size));


        // Allocate memory to store bucket counts
        size_t totalBucketSize = numBlocks * numBuckets * sizeof(unsigned int);
        unsigned int * h_bucketCount = (unsigned int *) malloc (numBuckets * sizeof (unsigned int));
        //array showing the number of elements in each bucket
        unsigned int * d_bucketCount;
        CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));

        // Allocate memory to store the new vector for kVals
        T * d_newvector;
        CUDA_CALL(cudaMalloc(&d_newvector, length * sizeof(T)));


        // array of kth buckets
        int numUniqueBuckets;
        int numUniqueBucketsOld;
        int lengthOld;
        int tempKorderLength;
        unsigned int * d_kVals;
        unsigned int * kthBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kthBuckets;
        unsigned int * kthBucketScanner = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kthBucketScanner;
        unsigned int * kIndices = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kIndices;
        unsigned int * uniqueBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_uniqueBuckets;
        unsigned int * uniqueBucketCounts = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_uniqueBucketCounts;
        unsigned int * reindexCounter = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_reindexCounter;
        unsigned int * kthnumBuckets = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_kthnumBuckets;
        T * tempOutput = (T *)malloc(numMemory * sizeof(T));
        T * d_tempOutput;
        unsigned int * tempKorderBucket = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_tempKorderBucket;
        unsigned int * tempKorderIndeces = (unsigned int *)malloc(numMemory * sizeof(unsigned int));
        unsigned int * d_tempKorderIndeces;
        CUDA_CALL(cudaMalloc(&d_kVals, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kIndices, numMemory * sizeof (unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kthBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kthBucketScanner, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_uniqueBucketCounts, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_reindexCounter, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_kthnumBuckets, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_tempOutput, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_tempKorderBucket, numMemory * sizeof(unsigned int)));
        CUDA_CALL(cudaMalloc(&d_tempKorderIndeces, numMemory * sizeof(unsigned int)));

        for (register int i = 0; i < numMemory; i++) {
            kthBucketScanner[i] = 0;
            kIndices[i] = i;
        }


        /// ***********************************************************
        /// **** STEP 1: Initialization
        /// **** STEP 1.3: Sort the klist
        /// **** and we have to keep the old index
        /// ***********************************************************

        CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, numKs * sizeof (unsigned int),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_kVals, kVals, numKs * sizeof (unsigned int),
                             cudaMemcpyHostToDevice));

        // sort the given indices
        thrust::device_ptr<unsigned int>kVals_ptr(d_kVals);
        thrust::device_ptr<unsigned int>kIndices_ptr(d_kIndices);
        thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);

        CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (unsigned int),
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(kVals, d_kVals, numKs * sizeof (unsigned int),
                             cudaMemcpyDeviceToHost));

        int kMaxIndex = numKs - 1;
        int kOffsetMax = 0;
        while (kVals[kMaxIndex] == length) {
            output[kIndices[numKs-1]] = maximum;
            numKs--;
            kMaxIndex--;
            kOffsetMax++;
        }

        int kOffsetMin = 0;
        while (kVals[0] == 1) {
            output[kIndices[0]] = minimum;
            kIndices++;
            kVals++;
            numKs--;
            kOffsetMin++;
        }


        /// ***********************************************************
        /// **** STEP 2: CreateBuckets
        /// ****  Declare and Generate Pivots and Slopes
        /// ***********************************************************

        // Find bucket sizes using a randomized selection
        generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize,
                          numBuckets, minimum, maximum);


        // make any slopes that were infinity due to division by zero (due to no
        //  difference between the two associated pivots) into zero, so all the
        //  values which use that slope are projected into a single bucket
        for (register int i = 0; i < numPivots - 1; i++)
            if (isinf(slopes[i]))
                slopes[i] = 0;



        // documentation
        for (int i = 0; i < numPivots - 1; i++) {
            pivotsLeft[i] = pivots[i];
            pivotsRight[i] = pivots[i + 1];
            kthnumBuckets[i] = numBuckets / (numPivots - 1) * i;
        }
        numUniqueBuckets = numPivots - 1;


        CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pivotsLeft, pivotsLeft, numUniqueBuckets * sizeof(T),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pivotsRight, pivotsRight, numUniqueBuckets * sizeof(T),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_kthnumBuckets, kthnumBuckets, numUniqueBuckets * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));



        /// ***********************************************************
        /// **** STEP 3: AssignBuckets
        /// **** Using the function assignSmartBucket
        /// ***********************************************************
        assignSmartBucket_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(T) +
                                      numUniqueBuckets * sizeof(double) + numUniqueBuckets * sizeof(unsigned int) +
                                      numBuckets * sizeof(unsigned int)>>>
                                      (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_pivotsRight,
                                       d_kthnumBuckets, d_bucketCount, numUniqueBuckets, numBuckets, offset);


        /// ***********************************************************
        /// **** STEP 4: IdentifyActiveBuckets
        /// **** Find the kth buckets
        /// **** and update their respective indices
        /// ***********************************************************

        sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

        findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, kthBucketScanner, kthBuckets, numBlocks);

        updatekVals_iterative<T>(kVals, &numKs, output, kIndices, &length, &lengthOld, h_bucketCount, kthBuckets, kthBucketScanner,
                              reindexCounter, uniqueBuckets, uniqueBucketCounts, &numUniqueBuckets, &numUniqueBucketsOld,
                              tempKorderBucket, tempKorderIndeces, &tempKorderLength);

        if (tempKorderLength > 0) {
            CUDA_CALL(cudaMemcpy(d_tempKorderBucket, tempKorderBucket, tempKorderLength * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_tempKorderIndeces, tempKorderIndeces, tempKorderLength * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));

            // potential to fix how many blocks to assign
            updateOutput_iterative<<<numBlocks, threadsPerBlock>>>(d_vector, d_elementToBucket, lengthOld, d_tempOutput,
                                                                   d_tempKorderBucket, tempKorderLength, offset);

            CUDA_CALL(cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(T),
                                 cudaMemcpyDeviceToHost));

            for (int i = 0; i < tempKorderLength; i++)
                output[tempKorderIndeces[i]] = tempOutput[i];

        }




        /// ***********************************************************
        /// **** STEP 5: Reduce
        /// **** Iteratively go through the loop to find correct
        /// **** order statistics and reduce the vector size
        /// ***********************************************************


        for (int j = 0; j < 4; j++) {

            printf("This is iteration %d.\n", j);

            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.1: Copy active elements
            /// **** Copy the elements from the unique active buckets
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_reindexCounter, reindexCounter,
                                 numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets,
                                 numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice));

            reindexCounts<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), threadsPerBlock>>>
                            (d_bucketCount, numBuckets, numBlocks, d_reindexCounter, d_uniqueBuckets, numUniqueBuckets);

            copyElements_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(unsigned int)>>>
                                (d_vector, d_newvector, lengthOld, d_elementToBucket, d_uniqueBuckets, numUniqueBuckets,
                                 d_bucketCount, numBuckets, offset);

            swapPointers(&d_vector, &d_newvector);




            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.2: Update the pivots
            /// **** Update pivots to generate Pivots and Slopes in Step 5.3
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, numUniqueBuckets * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));


            // potential to fix how many blocks to assign
            updatePivots_iterative<T><<<numBlocks, threadsPerBlock>>>(d_pivotsLeft, d_newPivotsLeft, d_newPivotsRight,
                                                                   d_slopes, d_kthnumBuckets, d_uniqueBuckets,
                                                                   numUniqueBuckets, numUniqueBucketsOld, offset);

            swapPointers(&d_pivotsLeft, &d_newPivotsLeft);
            swapPointers(&d_pivotsRight, &d_newPivotsRight);


            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.3: create slopes and buckets offset
            /// **** create slopes and buckets offset for next iteration
            /// ***********************************************************

            CUDA_CALL(cudaMemcpy(d_uniqueBucketCounts, uniqueBucketCounts, numUniqueBuckets * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));


            // potential to fix how many blocks to assign
            generateBucketsandSlopes_iterative<<<numBlocks, threadsPerBlock>>>
                                                 (d_pivotsLeft, d_pivotsRight, d_slopes, d_uniqueBucketCounts,
                                                  numUniqueBuckets, d_kthnumBuckets, length, offset, numBuckets);

            CUDA_CALL(cudaMemcpy(slopes, d_slopes, numUniqueBuckets * sizeof(double),
                                 cudaMemcpyDeviceToHost));

            // make any slopes that were infinity due to division by zero (due to no
            //  difference between the two associated pivots) into zero, so all the
            //  values which use that slope are projected into a single bucket
            for (register int i = 0; i < numUniqueBuckets; i++)
                if (isinf(slopes[i]))
                    slopes[i] = 0;

            CUDA_CALL(cudaMemcpy(d_slopes, slopes, numUniqueBuckets * sizeof(double),
                                 cudaMemcpyHostToDevice));


            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.4: assign buckets
            /// **** assign elements to correct buckets in iteration
            /// ***********************************************************
            assignSmartBucket_iterative<T><<<numBlocks, threadsPerBlock, numUniqueBuckets * sizeof(T) +
                                                                         numUniqueBuckets * sizeof(double) +
                                                                         numUniqueBuckets * sizeof(unsigned int) +
                                                                         numBuckets * sizeof(unsigned int)>>>
                    (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_pivotsRight, d_kthnumBuckets,
                     d_bucketCount, numUniqueBuckets, numBuckets, offset);


            /// ***********************************************************
            /// **** STEP 5: Reduce
            /// **** Step 5.5: IdentifyActiveBuckets
            /// **** Find kth buckets and update their respective indices
            /// ***********************************************************

            sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks);

            findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, kthBucketScanner, kthBuckets, numBlocks);

            updatekVals_iterative<T>(kVals, &numKs, output, kIndices, &length, &lengthOld, h_bucketCount, kthBuckets, kthBucketScanner,
                                     reindexCounter, uniqueBuckets, uniqueBucketCounts, &numUniqueBuckets, &numUniqueBucketsOld,
                                     tempKorderBucket, tempKorderIndeces, &tempKorderLength);

            if (tempKorderLength > 0) {
                CUDA_CALL(cudaMemcpy(d_tempKorderBucket, tempKorderBucket, tempKorderLength * sizeof(unsigned int),
                                     cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(d_tempKorderIndeces, tempKorderIndeces, tempKorderLength * sizeof(unsigned int),
                                     cudaMemcpyHostToDevice));

                // potential to fix how many blocks to assign
                updateOutput_iterative<<<numBlocks, threadsPerBlock>>>(d_vector, d_elementToBucket, lengthOld, d_tempOutput,
                                                                       d_tempKorderBucket, tempKorderLength, offset);

                CUDA_CALL(cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(T),
                                     cudaMemcpyDeviceToHost));

                for (int i = 0; i < tempKorderLength; i++)
                    output[tempKorderIndeces[i]] = tempOutput[i];

            }

            if (numKs <= 0)
                return 0;

        }


        CUDA_CALL(cudaMemcpy(d_kthBuckets, kthBuckets, numKs * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

        updateOutput_iterative<<<numBlocks, threadsPerBlock>>>(d_vector, d_elementToBucket, lengthOld, d_tempOutput,
                                                               d_kthBuckets, numKs, offset);

        CUDA_CALL(cudaMemcpy(tempOutput, d_tempOutput, numKs * sizeof(T),
                             cudaMemcpyDeviceToHost));

        for (int i = 0; i < numKs; i++)
            output[kIndices[i]] = tempOutput[i];

        /*
        int * h_vector = (int *)malloc(length * sizeof(int));
        cudaMemcpy(h_vector, d_vector, length * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < length; i++) {
            printf("%d   ", h_vector[i]);
        }

        printf("\n");
         */

	free(slopes);
	free(pivots);
	free(pivotsLeft);
	free(pivotsRight);
	free(newPivotsLeft);
	free(newPivotsRight);
	free(h_bucketCount);
	free(kthBuckets);
	free(kthBucketScanner);
	free(kIndices);
	free(uniqueBuckets);
	free(uniqueBucketCounts);
	free(reindexCounter);
	free(kthnumBuckets);
	free(tempOutput);
	free(tempKorderBucket);
	free(tempKorderIndeces);


	cudaFree(d_slopes);
	cudaFree(d_pivots);
	cudaFree(d_pivotsLeft);
	cudaFree(d_pivotsRight);
	cudaFree(d_newPivotsLeft);
	cudaFree(d_newPivotsRight);
	cudaFree(d_elementToBucket);
	cudaFree(d_bucketCount);
	cudaFree(d_newvector);
	cudaFree(d_kVals);
	cudaFree(d_kthBuckets);
	cudaFree(d_kthBucketScanner);
	cudaFree(d_kIndices);
	cudaFree(d_uniqueBuckets);
	cudaFree(d_uniqueBucketCounts);
	cudaFree(d_reindexCounter);
	cudaFree(d_kthnumBuckets);
	cudaFree(d_tempOutput);
	cudaFree(d_tempKorderBucket);
	cudaFree(d_tempKorderIndeces);

        return 0;
    }

    template <typename T>
    T iterativeSMOSWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
                              , T * outputs, int blocks, int threads) {

        int numBuckets = 8192;
	unsigned int * kVals = (unsigned int *)malloc(numKs * sizeof(unsigned int));

	// turn it into kth smallest
    	for (register int i = 0; i < numKs; i++) 
      	    kVals[i] = length - kVals_ori[i] + 1;

	iterativeSMOS(d_vector, length, kVals, numKs, outputs, blocks, threads, numBuckets, 17);
	
	free(kVals);

	return 1;
    }

}


/*
int main() {



    /*
    // Test for generatePivots_iterative
    int numUniqueBuckets = 10;
    int numTotalBuckets = 1000;
    int offset = 1024;
    int length = 10000;

    int* pivotsLeft = (int*)malloc(numUniqueBuckets * sizeof(int));
    int* pivotsRight = (int*)malloc(numUniqueBuckets * sizeof(int));
    for (int i = 0; i < 10; i++) {
        pivotsLeft[i] = i * 100;
        pivotsRight[i] = i * 100 + 100;
    }

    unsigned int* uniqueBucketsCounts = (unsigned int*)malloc(numUniqueBuckets * sizeof(unsigned int));
    for (int i = 0; i < 10; i++) {
        uniqueBucketsCounts[i] = 1000;
    }

    double* slopes = (double*)malloc(numUniqueBuckets * sizeof(double));

    unsigned int* kthnumBuckets = (unsigned int*)malloc(numUniqueBuckets * sizeof(unsigned int));


    int * d_pivotsLeft, * d_pivotsRight;
    double * d_slopes;
    unsigned int * d_uniqueBucketsCounts, * d_kthnumBuckets;
    cudaMalloc((void**)&d_pivotsLeft, numUniqueBuckets * sizeof(int));
    cudaMalloc((void**)&d_pivotsRight, numUniqueBuckets * sizeof(int));
    cudaMalloc((void**)&d_slopes, numUniqueBuckets * sizeof(double));
    cudaMalloc((void**)&d_uniqueBucketsCounts, numUniqueBuckets * sizeof(unsigned int));
    cudaMalloc((void**)&d_kthnumBuckets, numUniqueBuckets * sizeof(unsigned int));


    cudaMemcpy(d_pivotsLeft,pivotsLeft,numUniqueBuckets * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivotsRight,pivotsRight,numUniqueBuckets * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_uniqueBucketsCounts, uniqueBucketsCounts, numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice);


    dim3 dimBlock(1000,1,1);
    dim3 dimGrid(1000,1,1);

    generatePivots_iterative<<<dimGrid,dimBlock>>>(d_pivotsLeft,d_pivotsRight,d_slopes,d_uniqueBucketsCounts,numUniqueBuckets,
                                         d_kthnumBuckets,length,offset,numTotalBuckets);

    cudaMemcpy(slopes, d_slopes,numUniqueBuckets * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(kthnumBuckets, d_kthnumBuckets, numUniqueBuckets * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < numUniqueBuckets; i++) {
        printf("%d  ", kthnumBuckets[i]);
    }

    printf("\n");

    for (int i = 0; i < numUniqueBuckets; i++) {
        printf("%lf  ", slopes[i]);
    }

     */



    /*
    // Test for binary search inside the assignSmartBucket_iterative
    int num = 145;
    int numPivots = 7;
    int minPivotIndex = 0;
    int maxPivotIndex = numPivots - 1;
    int midPivotIndex;

    int pivotsLeft[7] = {0, 32, 70, 75, 97, 111, 140};

    if (num >= pivotsLeft[numPivots - 1]) {
        minPivotIndex = numPivots - 1;
    }
    else {
        for (int j = 1; j < numPivots - 1; j *= 2) {
            midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
            if (num >= pivotsLeft[midPivotIndex])
                minPivotIndex = midPivotIndex;
            else
                maxPivotIndex = midPivotIndex;
        }
    }

    printf("%d", minPivotIndex);

     */



    /*
    // Tests for assignBuckets_iterative
    int threadsPerBlock = 1024;
    int numBlocks = 10;
    int numUniqueBuckets = 10;
    int numTotalBuckets = 8192;
    int offset = threadsPerBlock * numBlocks;
    int length = 100000;
    int numBuckets = 8192;

    int * h_vector;
    h_vector = (int*)malloc(length * sizeof(int));
    for (int i = 0; i < 7000; i++)
        h_vector[i] = i / 70;
    for (int i = 7000; i < 12000; i++)
        h_vector[i] = 400 + (i - 7000) / 10;
    for (int i = 12000; i < 25000; i++)
        h_vector[i] = 1000 + (i - 12000) / 10;
    for (int i = 25000; i < 35000; i++)
        h_vector[i] = 2400 + (i - 25000) / 10;
    for (int i = 35000; i < 50000; i++)
        h_vector[i] = 4000 + (i - 35000) / 10;
    for (int i = 50000; i < 69000; i++)
        h_vector[i] = 6000 + (i - 50000) / 10;
    for (int i = 69000; i < 75000; i++)
        h_vector[i] = 8000 + (i - 69000) / 10;
    for (int i = 75000; i < 84000; i++)
        h_vector[i] = 9000 + (i - 75000) / 10;
    for (int i = 84000; i < 98000; i++)
        h_vector[i] = 11000 + (i - 84000) / 10;
    for (int i = 98000; i < 100000; i++)
        h_vector[i] = 13000 + (i - 98000) / 10;

    unsigned int * h_elementToBucket;
    h_elementToBucket = (unsigned int*)malloc(length * sizeof(unsigned int));

    int * pivotsLeft;
    int * pivotsRight;
    pivotsLeft = (int*)malloc(numUniqueBuckets * sizeof(int));
    pivotsRight = (int*)malloc(numUniqueBuckets * sizeof(int));
    pivotsLeft[0] = 0; pivotsLeft[1] = 400; pivotsLeft[2] = 1000; pivotsLeft[3] = 2400; pivotsLeft[4] = 4000;
    pivotsLeft[5] = 6000; pivotsLeft[6] = 8000; pivotsLeft[7] = 9000; pivotsLeft[8] = 11000; pivotsLeft[9] = 13000;
    pivotsRight[0] = 100; pivotsRight[1] = 900; pivotsRight [2] = 2300; pivotsRight[3] = 3400; pivotsRight[4] = 5500;
    pivotsRight [5] = 7900; pivotsRight[6] = 8600; pivotsRight [7] = 9900; pivotsRight[8] = 12400; pivotsRight[9] = 13200;

    unsigned int * uniqueBucketCounts;
    uniqueBucketCounts = (unsigned int*)malloc(numUniqueBuckets * sizeof(unsigned int));
    uniqueBucketCounts[0] = 7000; uniqueBucketCounts[1] = 5000; uniqueBucketCounts[2] = 13000; uniqueBucketCounts[3] = 10000;
    uniqueBucketCounts[4] = 15000; uniqueBucketCounts[5] = 19000; uniqueBucketCounts[6] = 6000; uniqueBucketCounts[7] = 9000;
    uniqueBucketCounts[8] = 14000; uniqueBucketCounts[9] = 2000;


    int * d_vector;
    unsigned int * d_elementToBucket;
    double * d_slopes;
    int * d_pivotsLeft;
    int * d_pivotsRight;
    unsigned int * d_kthNumBuckets;
    unsigned int * d_bucketCount;
    unsigned int * d_uniqueBucketCounts;
    cudaMalloc((void**)&d_vector, length * sizeof(int));
    cudaMalloc((void**)&d_elementToBucket, length * sizeof(unsigned int));
    cudaMalloc((void**)&d_slopes, numUniqueBuckets * sizeof(double));
    cudaMalloc((void**)&d_pivotsLeft, numUniqueBuckets * sizeof(int));
    cudaMalloc((void**)&d_pivotsRight, numUniqueBuckets * sizeof(int));
    cudaMalloc((void**)&d_kthNumBuckets, numUniqueBuckets * sizeof(unsigned int));
    cudaMalloc((void**)&d_bucketCount, numBlocks * numBuckets * sizeof(unsigned int));
    cudaMalloc((void**)&d_uniqueBucketCounts, numUniqueBuckets * sizeof(unsigned int));

    cudaMemcpy(d_vector, h_vector, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivotsLeft, pivotsLeft, numUniqueBuckets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivotsRight, pivotsRight, numUniqueBuckets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uniqueBucketCounts, uniqueBucketCounts, numUniqueBuckets * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 dimGrid(numBlocks,1,1);
    dim3 dimBlock(threadsPerBlock,1,1);

    generatePivots_iterative<int><<<dimGrid,dimBlock>>>(d_pivotsLeft,d_pivotsRight,d_slopes,d_uniqueBucketCounts,numUniqueBuckets,
                                                        d_kthNumBuckets,length,offset,numTotalBuckets);


    **
    assignSmartBucket_iterative<int><<<dimGrid, dimBlock, numUniqueBuckets * sizeof(int) + numUniqueBuckets * sizeof(double)
                                                         + numUniqueBuckets * sizeof(unsigned int) + numBuckets * sizeof(unsigned int)>>>
                                                         (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_kthNumBuckets,
                                                          d_bucketCount, d_uniqueBucketCounts, numUniqueBuckets, numBuckets, offset);




    assignSmartBucket_iterative<int><<<dimGrid, dimBlock, numUniqueBuckets * sizeof(int) + numUniqueBuckets * sizeof(double)
                                                          + numUniqueBuckets * sizeof(unsigned int) + numBuckets * sizeof(unsigned int)>>>
            (d_vector, length, d_elementToBucket, d_slopes, d_pivotsLeft, d_kthNumBuckets,
             d_bucketCount, d_uniqueBucketCounts, numUniqueBuckets, numBuckets, offset);


    cudaMemcpy(h_elementToBucket, d_elementToBucket, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1000; i++)
    printf("%d\n", h_elementToBucket[i]);

    */



    /*
    //tests for updatekVals_iterative
    unsigned int * kVals = (unsigned int *)malloc(10 * sizeof(unsigned int));
    kVals[0] = 3; kVals[1] = 7; kVals[2] = 17; kVals[3] = 20; kVals[4] = 24;
    int numKs = 5;

    int * output = (int *)malloc(10 * sizeof(int));

    unsigned int * kIndicies = (unsigned int *)malloc(10 * sizeof(unsigned int));
    kIndicies[0] = 0; kIndicies[1] = 1; kIndicies[2] = 2; kIndicies[3] = 3; kIndicies[4] = 4;
    int length = 5;
    int lengthOld = 5;

    unsigned int * h_bucketCount = (unsigned int *)malloc(10 * sizeof(unsigned int));
    h_bucketCount[0] = 10; h_bucketCount[1] = 5; h_bucketCount[2] = 1; h_bucketCount[3] = 10;
    unsigned int * markedBuckets = (unsigned int *)malloc(10 * sizeof(unsigned int));
    markedBuckets[0] = 0; markedBuckets[1] = 0; markedBuckets[2] = 2; markedBuckets[3] = 3; markedBuckets[4] = 3;
    unsigned int * kthBucketScanner = (unsigned int *)malloc(10 * sizeof(unsigned int));
    kthBucketScanner[0] = 0; kthBucketScanner[1] = 0; kthBucketScanner[2] = 15; kthBucketScanner[3] = 16;
    kthBucketScanner[4] = 16;

    int * pivotsLeft = (int *)malloc(10 * sizeof(int));
    int * pivotsRight = (int *)malloc(10 * sizeof(int));

    unsigned int * reindexCounter = (unsigned int *)malloc(10 * sizeof(unsigned int));
    unsigned int * uniqueBuckets = (unsigned int *)malloc(10 * sizeof(unsigned int));
    unsigned int * uniqueBucketCounts = (unsigned int *) malloc(10 * sizeof(unsigned int));

    int numUniqueBuckets = 10;
    int numUniqueBucketsOld = 10;

    IterativeSMOS::updatekVals_iterative(kVals, &numKs, output, kIndicies, &length, &lengthOld, h_bucketCount, markedBuckets,
                                         kthBucketScanner, reindexCounter, uniqueBuckets, uniqueBucketCounts,
                                         &numUniqueBuckets, &numUniqueBucketsOld);

    for (int i = 0; i < numKs; i++)
        printf("%d  ", kVals[i]);

    printf("\n");

    for (int i = 0; i < numUniqueBuckets; i++)
        printf("%d  ", uniqueBuckets[i]);


    //IterativeSMOS::updatekVals_iterative(kVals, &numKs, output, kIndicies, &length, h_bucketCount, markedBuckets, kthBucketScanner,
            //         pivotsLeft, pivotsRight, reindexCounter, uniqueBuckets, &numUniqueBuckets);


     */



    /*
    // tests for updatePivots_iterative
    int * pivotsLeft = (int *)malloc(10 * sizeof(int));
    int * pivotsRight = (int *)malloc(10 * sizeof(int));
    pivotsLeft[0] = 0; pivotsLeft[1] = 100; pivotsLeft[2] = 200; pivotsLeft[3] = 300; pivotsLeft[4] = 400;
    pivotsRight[0] = 100; pivotsRight[1] = 200; pivotsRight[2] = 300; pivotsRight[3] = 400; pivotsRight[5] = 500;

    double * slopes = (double *)malloc(10 * sizeof(int));
    slopes[0] = 0.1; slopes[1] = 0.1; slopes[2] = 0.1; slopes[3] = 0.1; slopes[4] = 0.1;

    unsigned int * kthnumBuckets = (unsigned int *)malloc(10 * sizeof(unsigned int));
    kthnumBuckets[0] = 0; kthnumBuckets[1] = 10; kthnumBuckets[2] = 20; kthnumBuckets[3] = 30; kthnumBuckets[4] = 40;

    unsigned int * uniqueBuckets = (unsigned int *)malloc(10 * sizeof(unsigned int));
    uniqueBuckets[0] = 3; uniqueBuckets[1] = 21; uniqueBuckets[2] = 45;

    int numUniqueBuckets = 3;
    int numUniqueBucketsOld = 5;
    int threadPerBlock = 1024;
    int numBlock = 10;
    int offset = threadPerBlock * numBlock;

    int * d_pivotsLeft, * d_pivotsRight, * d_newPivotsLeft, * d_newPivotsRight;
    cudaMalloc((void**)&d_pivotsLeft, 10 * sizeof(int));
    cudaMalloc((void**)&d_pivotsRight, 10 * sizeof(int));
    cudaMalloc((void**)&d_newPivotsLeft, 10 * sizeof(int));
    cudaMalloc((void**)&d_newPivotsRight, 10 * sizeof(int));

    double * d_slopes;
    cudaMalloc((void**)&d_slopes, 10 * sizeof(double));

    unsigned int * d_kthnumBuckets, * d_uniqueBuckets;
    cudaMalloc((void**)&d_kthnumBuckets, 10 * sizeof(unsigned int));
    cudaMalloc((void**)&d_uniqueBuckets, 10 * sizeof(unsigned int));

    cudaMemcpy(d_pivotsLeft, pivotsLeft, 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivotsRight, pivotsRight, 10 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_slopes, slopes, 10 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kthnumBuckets, kthnumBuckets, 10 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uniqueBuckets, uniqueBuckets, 10 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int * outputpivotsLeft = (int *)malloc(10 * sizeof(int));
    int * outputpivotsRight = (int *)malloc(10 * sizeof(int));

    dim3 dimGrid(numBlock, 1, 1);
    dim3 dimBlock(threadPerBlock, 1, 1);

    updatePivots_iterative<<<dimGrid, dimBlock>>>(d_pivotsLeft, d_pivotsRight, d_newPivotsLeft, d_newPivotsRight,
                                                       d_slopes, d_kthnumBuckets, d_uniqueBuckets, numUniqueBuckets,
                                                       numUniqueBucketsOld, offset);

    swapPointers(&d_pivotsLeft, &d_newPivotsLeft);
    swapPointers(&d_pivotsRight, &d_newPivotsRight);

    cudaMemcpy(outputpivotsLeft, d_pivotsLeft, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputpivotsRight, d_pivotsRight, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numUniqueBuckets; i++) {
        printf("%d   %d\n", outputpivotsLeft[i], outputpivotsRight[i]);
    }

     */



    /*
    // test for updateOutput_iterative
    int lengthOld = 100000;
    int * vector = (int *)malloc(lengthOld * sizeof(int));
    unsigned int * elementToBucket = (unsigned int *)malloc(lengthOld * sizeof(unsigned int));
    int * d_vector;
    unsigned int * d_elementToBucket;
    cudaMalloc(&d_vector, lengthOld * sizeof(int));
    cudaMalloc(&d_elementToBucket, lengthOld * sizeof(unsigned int));

    int tempKorderLength = 10;
    int * tempOutput = (int *)malloc(tempKorderLength * sizeof(int));
    unsigned int * tempKorderBucket = (unsigned int *)malloc(tempKorderLength * sizeof(int));
    int * d_tempOutput;
    unsigned int * d_tempKorderBucket;
    cudaMalloc(&d_tempOutput, tempKorderLength * sizeof(int));
    cudaMalloc(&d_tempKorderBucket, tempKorderLength * sizeof(unsigned int));

    for (int i = 0; i < lengthOld; i++) {
        vector[i] = i + 100000;
        elementToBucket[i] = i;
    }

    tempKorderBucket[0] = 100; tempKorderBucket[1] = 1000; tempKorderBucket[2] = 10000;
    tempKorderBucket[3] = 20000; tempKorderBucket[4] = 30000; tempKorderBucket[5] = 40000;
    tempKorderBucket[6] = 50000; tempKorderBucket[7] = 60000; tempKorderBucket[8] = 70000;
    tempKorderBucket[9] = 80000;

    cudaMemcpy(d_vector, vector, lengthOld * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elementToBucket, elementToBucket, lengthOld * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempKorderBucket, tempKorderBucket, tempKorderLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int numBlocks = 10;
    int threadsPerBlock = 1024;
    int offset = numBlocks * threadsPerBlock;

    IterativeSMOS::updateOutput_iterative<<<numBlocks, threadsPerBlock>>>(d_vector, d_elementToBucket, lengthOld,
                                                                          d_tempOutput, d_tempKorderBucket,
                                                                          tempKorderLength, offset);

    cudaMemcpy(tempOutput, d_tempOutput, tempKorderLength * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < tempKorderLength; i++)
        printf("%d  ", tempOutput[i]);

     */



    /*
    // test for iterativeSMOS
    int threadsPerBlock = 1024;
    int numBlocks = 10;
    int numTotalBuckets = 8192;
    int offset = threadsPerBlock * numBlocks;
    int length = 100000;
    int numBuckets = 8192;

    int h_vector[100000];
    for (int i = 0; i < 50000; i++) {
        h_vector[i] = i * 2;
        h_vector[i+50000] = i * 2;
    }
    int * d_vector;
    cudaMalloc(&d_vector, 100000 * sizeof(int));
    cudaMemcpy(d_vector, h_vector, 100000 * sizeof(int), cudaMemcpyHostToDevice);

    unsigned int kVals[10] = {10, 1000, 10000, 40000, 50000, 60000, 70000, 80000, 90000, 5000};
    int numKs = 10;

    int * output = (int*)malloc(10 * sizeof(int));

    for (int i = 0; i < 10; i++) {
        output[i] = 0;
    }

    IterativeSMOS::iterativeSMOS(d_vector, length,kVals, numKs, output, numBlocks, threadsPerBlock, numTotalBuckets, 17);

    for (int i = 0; i < 10; i++) {
        printf("%d   ", output[i]);
    }
    




    return 0;
   
}
*/







