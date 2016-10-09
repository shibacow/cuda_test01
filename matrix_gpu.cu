#include <stdio.h>
#include <stdlib.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

#define MATRIX_SIZE 1024/*行列１辺の数*/
#define BLOCK_SIZE 16

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv){
  unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

  int* hMatrixA;
  int* hMatrixB;
  int* hMatrixC;
  hMatrixA = (int*)malloc(matrixSize);
  hMatrixB = (int*)malloc(matrixSize);

  /*初期値設定*/
  unsigned int col_idx, row_idx;
  for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++){
    for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++){
      hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
      hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
    }
  }

  /*デバイス側の変数設定*/
  int* dMatrixA;
  int* dMatrixB;
  int* dMatrixC;

  /*デバイスメモリ領域の確保*/
  checkCudaErrors(cudaMalloc((void**)&dMatrixA, matrixSize));
  checkCudaErrors(cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&dMatrixB, matrixSize));
  checkCudaErrors(cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&dMatrixC, matrixSize));

  /*ブロックサイズとグリッドサイズの設定*/
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);

  /*タイマーを作成して計測開始*/
  cudaevent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start, NULL)); // スタート

  /*カーネルの起動*/
  matrixMul<<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);
  cudaThreadSynchronize();

  /*結果の領域確保とデバイス側からのメモリ転送*/
  hMatrixC = (int*)malloc(matrixSize);
  checkCudaErrors(cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost));

  /*タイマーを停止しかかった時間を表示*/

  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("Processing time: %f (msec)\n", msecTotal);

  /*ホスト・デバイスメモリの開放*/
  free(hMatrixA);
  free(hMatrixB);
  free(hMatrixC);
  checkCudaErrors(cudaFree(dMatrixA));
  checkCudaErrors(cudaFree(dMatrixB));
  checkCudaErrors(cudaFree(dMatrixC));

  /*終了処理*/
  cudaThreadExit();
  exit(1);
}

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC){
  unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int scan_idx;
  unsigned int target = 0;

  /*行列の演算を行う*/
  for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
    target +=inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
    __syncthreads();
  }
  inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
