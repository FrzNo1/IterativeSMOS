/* Copyright 2012 Jeffrey Blanchard, Erik Opavsky, and Emircan Uysaler
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <limits>

namespace BucketMultiselect{
  using namespace std;

#define MAX_THREADS_PER_BLOCK 1024
#define CUTOFF_POINT 200000 

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER CPU FUNCTIONS
  /// ***********************************************************
  /// ***********************************************************



  /* This timing function uses CUDA event timing to process the amount of time
     required, and print out result with the given index.

     start a timer with option = 0
     stop a timer with option = 1
  */

  cudaEvent_t start, stop;
  float time;

  void timing(int option, int ind){
    if(option == 0) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
    }
    else {
      cudaThreadSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      printf("Time %d: %lf \n", ind, time);
    }
  }



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
  inline int findKBuckets(uint * d_bucketCount, uint * h_bucketCount, int numBuckets
                          , uint * kVals, int numKs, uint * sums, uint * markedBuckets
                          , int numBlocks) {
    // consider the last row which holds the total counts
    int sumsRowIndex= numBuckets * (numBlocks-1);

    CUDA_CALL(cudaMemcpy(h_bucketCount, d_bucketCount + sumsRowIndex, 
                         sizeof(uint) * numBuckets, cudaMemcpyDeviceToHost));

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


  /// ***********************************************************
  /// ***********************************************************
  /// **** HELPER GPU FUNCTIONS-KERNELS
  /// ***********************************************************
  /// ***********************************************************



  /* This function assigns elements to buckets based on the pivots and slopes determined 
     by a randomized sampling of the elements in the vector. At the same time, this 
     function keeps track of count.

     d_elementToBucket : bucket assignment for every array element
     d_bucketCount : number of element that falls into the indexed buckets within the block
  */
  template <typename T>
  __global__ void assignSmartBucket (T * d_vector, int length, int numBuckets
                                     , double * slopes, T * pivots, int numPivots
                                     , uint* d_elementToBucket , uint* d_bucketCount, int offset) {
  
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    uint bucketIndex;
    int threadIndex = threadIdx.x;  
    
    //variables in shared memory for fast access
    __shared__ int sharedNumSmallBuckets;
    if (threadIndex < 1) 
      sharedNumSmallBuckets = numBuckets / (numPivots-1);
    
    extern __shared__ uint array[];
    double * sharedSlopes = (double *)array;
    T * sharedPivots = (T *)&sharedSlopes[numPivots-1];
    uint * sharedBuckets = (uint *)&sharedPivots[numPivots];
  
    //reading bucket counts into shared memory where increments will be performed
    for (int i = 0; i < (numBuckets / MAX_THREADS_PER_BLOCK); i++) 
      if (threadIndex < numBuckets) 
        sharedBuckets[i * MAX_THREADS_PER_BLOCK + threadIndex] = 0;

    if(threadIndex < numPivots) {
      *(sharedPivots + threadIndex) = *(pivots + threadIndex);
      if(threadIndex < numPivots-1) 
        sharedSlopes[threadIndex] = slopes[threadIndex];
    }

    __syncthreads();

    //assigning elements to buckets and incrementing the bucket counts
    if(index < length) {
      int i;

      for(i = index; i < length; i += offset) {
        T num = d_vector[i];
        int minPivotIndex = 0;
        int maxPivotIndex = numPivots-1;
        int midPivotIndex;

        // find the index of the pivot that is the greatest s.t. lower than or equal to 
        // num using binary search
        for(int j = 1; j < numPivots - 1; j*=2) {
          midPivotIndex = (maxPivotIndex + minPivotIndex) / 2;
          if (num >= sharedPivots[midPivotIndex])
            minPivotIndex = midPivotIndex;
          else
            maxPivotIndex = midPivotIndex;
        }

        //bucketIndex = (minPivotIndex * sharedNumSmallBuckets) 
        //+ (int) (((double)num - (double)sharedPivots[minPivotIndex]) 
        //* sharedSlopes[minPivotIndex]);
        bucketIndex = (minPivotIndex * sharedNumSmallBuckets) 
          + (int) (((double)num - (double)sharedPivots[minPivotIndex]) 
                   * sharedSlopes[minPivotIndex]);

        if (bucketIndex == numBuckets) 
          bucketIndex= numBuckets-1;

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
  __global__ void sumCounts(uint * d_bucketCount, const int numBuckets
                            , const int numBlocks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=1; j<numBlocks; j++) 
      d_bucketCount[index + numBuckets*j] += d_bucketCount[index + numBuckets*(j-1)];
    
  }



  /* This function reindexes the buckets counts for every block according to the 
     accumulated d_reindexCounter counter for the reduced vector.
     updates d_bucketCount
  */
  __global__ void reindexCounts(uint * d_bucketCount, const int numBuckets
                                , const int numBlocks, uint * d_reindexCounter
                                , uint * d_markedBuckets , const int numUniqueBuckets) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex<numUniqueBuckets) {
      int index = d_markedBuckets[threadIndex];
      uint add = d_reindexCounter[threadIndex];

      for(int j=0; j<numBlocks; j++) 
        d_bucketCount[index + numBuckets*j] += add;
    }
  }



  /* This function copies the elements of buckets that contain kVals into a newly allocated 
     reduced vector space.
     newArray - reduced size vector containing the essential elements
  */
  template <typename T>
  __global__ void copyElements (T* d_vector, int length, uint* elementToBucket
                                , uint * buckets, const int numBuckets, T* newArray, uint offset
                                , uint * d_bucketCount, int numTotalBuckets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex;
    int loop = numBuckets / MAX_THREADS_PER_BLOCK;

    //extern __shared__ uint array[];
    //uint * sharedBucketCounts= (uint*)array;
    //uint * sharedBuckets= (uint*)&sharedBucketCounts[numBuckets];
    extern __shared__ uint sharedBuckets[];
    //uint * sharedBuckets= (uint*)&array;

    for (int i = 0; i <= loop; i++) {      
      threadIndex = i * blockDim.x + threadIdx.x;
      if(threadIndex < numBuckets) {
        sharedBuckets[threadIndex] = buckets[threadIndex];
        /*
        sharedBucketCounts[threadIndex] = 
          d_bucketCount[blockIdx.x * numTotalBuckets + sharedBuckets[threadIndex]];
        */
      }
    }
    
    __syncthreads();

    int minBucketIndex;
    int maxBucketIndex; 
    int midBucketIndex;
    //uint holder;
    uint temp;

    if(idx < length) {
      for(int i=idx; i<length; i+=offset) {
        temp = elementToBucket[i];
        minBucketIndex = 0;
        maxBucketIndex = numBuckets-1;

        //binary search over the markedBuckets to find a match quickly
        for(int j = 1; j < numBuckets; j*=2) {  
          midBucketIndex = (maxBucketIndex + minBucketIndex) / 2;
          if (temp > sharedBuckets[midBucketIndex])
            minBucketIndex=midBucketIndex+1;
          else
            maxBucketIndex=midBucketIndex;
        }

        if (buckets[maxBucketIndex] == temp) 
          /*
          newArray[atomicDec(sharedBucketCounts + maxBucketIndex, length)-1] = 
            d_vector[i];
          */
          newArray[atomicDec(d_bucketCount + blockIdx.x * numTotalBuckets 
                             + sharedBuckets[maxBucketIndex], length)-1] = d_vector[i];
      }
    }

  }



  /* This function speeds up the copying process the requested kVals by clustering them
     together.
  */
  template <typename T>
  __global__ void copyValuesInChunk (T * outputVector, T * inputVector, uint * kList
                                     , uint * kIndices, int kListCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int loop = kListCount / MAX_THREADS_PER_BLOCK;

    for (int i = 0; i <= loop; i++) {      
      if (idx < kListCount)
        *(outputVector + *(kIndices + idx)) = *(inputVector + *(kList + idx) - 1);
    }
  }


  /// ***********************************************************
  /// ***********************************************************
  /// **** GENERATE PIVOTS
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
    uint seed;

    gettimeofday(&t1, NULL);
    seed = t1.tv_usec * t1.tv_sec;
  
    thrust::device_ptr<T> d_ptr(d_vec);
    thrust::transform (thrust::counting_iterator<uint>(0), 
                       thrust::counting_iterator<uint>(size), 
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

  __global__ void enlargeIndexAndGetElements (float * in, uint * out, uint * list, int size) {
    *(out + blockIdx.x * blockDim.x + threadIdx.x) = 
      (uint) *(list + ((int) (*(in + blockIdx.x * blockDim.x + threadIdx.x) * size)));
  }



  /* This function generates Pivots from the random sampled data and calculates slopes.
 
     pivots - arrays of pivots
     slopes - array of slopes
  */
  template <typename T>
  void generatePivots (uint * pivots, double * slopes, uint * d_list, int sizeOfVector
, int numPivots, int sizeOfSample, int totalSmallBuckets, uint min, uint max) {
  
    float * d_randomFloats;
    uint * d_randomInts;
    int endOffset = 22;
    int pivotOffset = (sizeOfSample - endOffset * 2) / (numPivots - 3);
    int numSmallBuckets = totalSmallBuckets / (numPivots - 1);

    cudaMalloc (&d_randomFloats, sizeof (float) * sizeOfSample);
  
    d_randomInts = (uint *) d_randomFloats;

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
    cudaMemcpy (pivots + 1, d_randomInts + endOffset - 1, sizeof (uint)
                , cudaMemcpyDeviceToHost);
    cudaMemcpy (pivots + numPivots - 2, d_randomInts + sizeOfSample - endOffset - 1, 
                sizeof (uint), cudaMemcpyDeviceToHost);
    slopes[0] = numSmallBuckets / (double) (pivots[1] - pivots[0]);

    for (register int i = 2; i < numPivots - 2; i++) {
      cudaMemcpy (pivots + i, d_randomInts + pivotOffset * (i - 1) + endOffset - 1, 
                  sizeof (uint), cudaMemcpyDeviceToHost);
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
  /// **** bucketMultiSelect: the main algorithm
  /// ***********************************************************
  /// ***********************************************************

  /* This function is the main process of the algorithm. It reduces the given multi-selection
     problem to a smaller problem by using bucketing ideas.
  */
  template <typename T>
  T bucketMultiSelect (T* d_vector, int length, uint * kVals, int numKs, T * output, int blocks
              , int threads, int numBuckets, int numPivots) {    

    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.1: Find Min and Max of the whole vector
    /// **** We don't need to go through the rest of the algorithm if it's flat
    /// ***********************************************************
    // timing(0, 1);

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

    // timing(1, 1);
    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.2: Declare variables and allocate memory
    /// **** Declare Variables
    /// ***********************************************************
    // timing(0, 2);

    //declaring variables for kernel launches
    int threadsPerBlock = threads;
    int numBlocks = blocks;
    int offset = blocks * threads;

    // variables for the randomized selection
    int sampleSize = 1024;

    // pivot variables
    double slopes[numPivots - 1];
    double * d_slopes;
    T pivots[numPivots];
    T * d_pivots;

    //Allocate memory to store bucket assignments
    size_t size = length * sizeof(uint);
    uint * d_elementToBucket;    //array showing what bucket every element is in

    CUDA_CALL(cudaMalloc(&d_elementToBucket, size));

    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBlocks * numBuckets * sizeof(uint);
    uint * h_bucketCount = (uint *) malloc (numBuckets * sizeof (uint));
    //array showing the number of elements in each bucket
    uint * d_bucketCount; 

    CUDA_CALL(cudaMalloc(&d_bucketCount, totalBucketSize));

    // array of kth buckets
    int numUniqueBuckets;
    uint * d_kVals; 
    uint kthBuckets[numKs]; 
    uint kthBucketScanner[numKs]; 
    uint * kIndices = (uint *) malloc (numKs * sizeof (uint));
    uint * d_kIndices;
    uint uniqueBuckets[numKs];
    uint * d_uniqueBuckets; 
    uint reindexCounter[numKs];  
    uint * d_reindexCounter;    

    CUDA_CALL(cudaMalloc(&d_kVals, numKs * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_kIndices, numKs * sizeof (uint)));

    for (register int i = 0; i < numKs; i++) {
      kthBucketScanner[i] = 0;
      kIndices[i] = i;
    }

    // variable to store the end result
    int newInputLength;
    T* newInput;

    // timing(1, 2);
    /// ***********************************************************
    /// **** STEP 1: Initialization 
    /// **** STEP 1.3: Sort the klist
    /// and keep the old index
    /// ***********************************************************
    // timing(0, 3);

    CUDA_CALL(cudaMemcpy(d_kIndices, kIndices, numKs * sizeof (uint), 
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kVals, kVals, numKs * sizeof (uint), 
                         cudaMemcpyHostToDevice)); 

    // sort the given indices
    thrust::device_ptr<uint>kVals_ptr(d_kVals);
    thrust::device_ptr<uint>kIndices_ptr(d_kIndices);
    thrust::sort_by_key(kVals_ptr, kVals_ptr + numKs, kIndices_ptr);

    CUDA_CALL(cudaMemcpy(kIndices, d_kIndices, numKs * sizeof (uint), 
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(kVals, d_kVals, numKs * sizeof (uint), 
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

    // timing(1, 3);
    /// ***********************************************************
    /// **** STEP 2: CreateBuckets 
    /// ****  Declare and Generate Pivots and Slopes
    /// ***********************************************************
    // timing(0, 4);

    CUDA_CALL(cudaMalloc(&d_slopes, (numPivots - 1) * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_pivots, numPivots * sizeof(T)));

    // Find bucket sizes using a randomized selection
    generatePivots<T>(pivots, slopes, d_vector, length, numPivots, sampleSize, 
                      numBuckets, minimum, maximum);
    
    // make any slopes that were infinity due to division by zero (due to no 
    //  difference between the two associated pivots) into zero, so all the
    //  values which use that slope are projected into a single bucket
    for (register int i = 0; i < numPivots - 1; i++)
      if (isinf(slopes[i]))
        slopes[i] = 0;

    /*
    for (register int i = 0; i < numPivots; i++)
      printf("piv %lf \n", pivots[i]);
    for (register int i = 0; i < numPivots - 1; i++)
      printf("%lf \n", slopes[i]);
    */

    CUDA_CALL(cudaMemcpy(d_slopes, slopes, (numPivots - 1) * sizeof(double), 
                         cudaMemcpyHostToDevice));  
    CUDA_CALL(cudaMemcpy(d_pivots, pivots, numPivots* sizeof(T), 
                         cudaMemcpyHostToDevice));

    // timing(1, 4);
    /// ***********************************************************
    /// **** STEP 3: AssignBuckets 
    /// **** Using the function assignSmartBucket
    /// ***********************************************************
    // timing(0, 5);

    //Distribute elements into their respective buckets
    assignSmartBucket<T><<<numBlocks, threadsPerBlock, numPivots * sizeof(T) 
      + (numPivots-1) * sizeof(double) + numBuckets * sizeof(uint)>>>
      (d_vector, length, numBuckets, d_slopes, d_pivots, numPivots, 
       d_elementToBucket, d_bucketCount, offset);
    // timing(1, 5);

    /// ***********************************************************
    /// **** STEP 4: IdentifyActiveBuckets 
    /// **** Find the kth buckets
    /// **** and update their respective indices
    /// ***********************************************************
    // timing(0, 6);

    sumCounts<<<numBuckets/threadsPerBlock, threadsPerBlock>>>(d_bucketCount, 
                                                               numBuckets, numBlocks);


    findKBuckets(d_bucketCount, h_bucketCount, numBuckets, kVals, numKs, 
                 kthBucketScanner, kthBuckets, numBlocks);

    // timing(1, 6);
    // timing(0, 7);

    // we must update K since we have reduced the problem size to elements in the 
    // kth bucket.
    //  get the index of the first element
    //  add the number of elements
    uniqueBuckets[0] = kthBuckets[0];
    reindexCounter[0] = 0;
    numUniqueBuckets = 1;
    kVals[0] -= kthBucketScanner[0];

    for (int i = 1; i < numKs; i++) {
      if (kthBuckets[i] != kthBuckets[i-1]) {
        uniqueBuckets[numUniqueBuckets] = kthBuckets[i];
        reindexCounter[numUniqueBuckets] = 
          reindexCounter[numUniqueBuckets-1]  + h_bucketCount[kthBuckets[i-1]];
        numUniqueBuckets++;
      }
      kVals[i] = reindexCounter[numUniqueBuckets-1] + kVals[i] - kthBucketScanner[i];
    }

    newInputLength = reindexCounter[numUniqueBuckets-1] 
      + h_bucketCount[kthBuckets[numKs - 1]];
    /*
    if(numUniqueBuckets> 6*1024) {
       printf("bucketMultiselect isn't really advantageous on this size of data as it needs to use more than the available shared memory. Use sort&choose instead.\n");
       exit(0);
    }
    */
       
    //printf("bucketmultiselectBlocked total kbucket_count = %d\n", newInputLength);
    //printf("numMarkedBuckets = %d\n", numUniqueBuckets);
    //printf("reindex block = %d\n", (int) ceil((float)numUniqueBuckets/threadsPerBlock));

    // timing(1, 7);
    // timing(0, 22);

    // reindex the counts
    CUDA_CALL(cudaMalloc(&d_reindexCounter, numUniqueBuckets * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_uniqueBuckets, numUniqueBuckets * sizeof(uint)));

    CUDA_CALL(cudaMemcpy(d_reindexCounter, reindexCounter, 
                         numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_uniqueBuckets, uniqueBuckets, 
                         numUniqueBuckets * sizeof(uint), cudaMemcpyHostToDevice));

    reindexCounts<<<(int) ceil((float)numUniqueBuckets/threadsPerBlock), 
      threadsPerBlock>>>(d_bucketCount, numBuckets, numBlocks, d_reindexCounter, 
                         d_uniqueBuckets, numUniqueBuckets);

    // timing(1, 22);
    /// ***********************************************************
    /// **** STEP 5: Reduce 
    /// **** Copy the elements from the unique acitve buckets
    /// **** to a new vector 
    /// ***********************************************************
    // timing(0, 8);

    // allocate memory foir the new array
    CUDA_CALL(cudaMalloc(&newInput, newInputLength * sizeof(T)));
   
    // timing(1, 8);

    // timing(0, 9);
    /*
    copyElements<T><<<numBlocks, threadsPerBlock, 
      numUniqueBuckets * 2 * sizeof(uint)>>>(d_vector, length, d_elementToBucket, 
                                             d_uniqueBuckets, numUniqueBuckets, newInput, offset, 
                                             d_bucketCount, numBuckets);
    */
    copyElements<T><<<numBlocks, threadsPerBlock, 
      numUniqueBuckets * sizeof(uint)>>>(d_vector, length, d_elementToBucket, 
                                             d_uniqueBuckets, numUniqueBuckets, newInput, offset, 
                                             d_bucketCount, numBuckets);
  
    // timing(1, 9);

    /// ***********************************************************
    /// **** STEP 6: sort&choose
    /// **** Using thrust::sort on the reduced vector and the
    /// **** updated indices of the order statistics, 
    /// **** we solve the reduced problem.
    /// ***********************************************************

    //free all used memory
    cudaFree(d_pivots);
    cudaFree(d_slopes);  
    free(h_bucketCount); 
    cudaFree(d_bucketCount); 
    cudaFree(d_uniqueBuckets); 
    cudaFree(d_reindexCounter);  

    // timing(0, 10);
    // sort the vector
    thrust::device_ptr<T>newInput_ptr(newInput);
    thrust::sort(newInput_ptr, newInput_ptr + newInputLength);

    /* prepare for the new strategy for copying k values back in a chunk */
    
    T * d_output = (T *) d_elementToBucket;
    CUDA_CALL(cudaMemcpy (d_output, output, 
                          (numKs + kOffsetMin + kOffsetMax) * sizeof (T), 
                          cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy (d_kVals, kVals, numKs * sizeof (uint), 
                          cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy (d_kIndices, kIndices, numKs * sizeof (uint), 
                          cudaMemcpyHostToDevice));

    copyValuesInChunk<T><<<numBlocks, threadsPerBlock>>>(d_output, 
                                                         newInput, d_kVals, d_kIndices, numKs);

    CUDA_CALL(cudaMemcpy (output, d_output, 
                          (numKs + kOffsetMin + kOffsetMax) * sizeof (T), 
                          cudaMemcpyDeviceToHost));

    cudaFree(d_elementToBucket);  
    cudaFree(d_kIndices); 
    cudaFree(d_kVals); 
    
    /* done new strategy for copying k values back in a chunk */

    // timing(1, 10);
    // printf("finito\n");

    cudaFree(newInput); 
    free (kIndices - kOffsetMin);

    return 1;
  }

  /* Wrapper function around the multi-selection fucntion that inverts the given k indices.
   */
  template <typename T>
  T bucketMultiselectWrapper (T * d_vector, int length, uint * kVals_ori, int numKs
                              , T * outputs, int blocks, int threads) { 

    int numBuckets = 8192;
    uint kVals[numKs];

    // turn it into kth smallest
    for (register int i = 0; i < numKs; i++) 
      kVals[i] = length - kVals_ori[i] + 1;
    
    // empirically found cutoff points
    /*
      if (length <= 524288)
      numBuckets = 4096;
      else if (length <= 1048576 && numKs <= 188)
      numBuckets = 4096;
      else if (length <= 2097152 && numKs <= 94)
      numBuckets = 4096;
      else if (length <= 4194304 && numKs <= 48)
      numBuckets = 4096;
      else if (length <= 8388608 && numKs <= 20)
      numBuckets = 4096;
      else if (length <= 16777216 && numKs <= 10)
      numBuckets = 4096;
      else if (length <= 33554432 && numKs <= 6)
      numBuckets = 4096;
      else if (length <= 67108864 && numKs <= 4)
      numBuckets = 4096;
    */
   
    bucketMultiSelect (d_vector, length, kVals, numKs, outputs, blocks, threads, numBuckets, 17);

    return 1;
  }
}

