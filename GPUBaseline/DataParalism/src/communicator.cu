#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include "cublas_v2.h"
#include <limits> 
#define THREADNUM 1024

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
      printf("Failed: MPI error %s:%d '%d'\n",        \
          __FILE__,__LINE__, e);   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  
  #define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
  
  #define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
      printf("Failed, NCCL error %s:%d '%s'\n",             \
          __FILE__,__LINE__,ncclGetErrorString(r));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  

typedef void* stream_handle;

class ncclpack{
public:
  ncclpack(int myRank, int nRanks): myRank_(myRank), nRanks_(nRanks){
  }
  void nccl_init(){
    if (myRank_ == 0) ncclGetUniqueId(&id_);
    MPICHECK(MPI_Bcast((void *)&id_, sizeof(id_), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&comm_, nRanks_, id_, myRank_));
  }
  void nccl_reduce(float* sendbuff, float* recvbuff, size_t count, int root, cudaStream_t stream){
    NCCLCHECK(ncclReduce((const void*)sendbuff, (void*)recvbuff, count, ncclFloat, ncclSum, root, comm_, stream));
  }
  void nccl_broadcast(float* sendbuff, float* recvbuff, size_t count, int root, cudaStream_t stream){
    NCCLCHECK(ncclBroadcast((const void*)sendbuff, (void*)recvbuff, count, ncclFloat, root, comm_, stream));
  }
  void nccl_allgather(float* sendbuff, float* recvbuff, size_t count, cudaStream_t stream){
    NCCLCHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, count, ncclFloat, comm_, stream));
  }
  void nccl_allreduce(float* sendbuff, float* recvbuff, size_t count, cudaStream_t stream){
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, count, ncclFloat, ncclSum, comm_, stream));
  }
//   void nccl_send(){
//     NCCLCHECK(ncclSend((const void*)sendbuff, size_t count, ncclFloat, peer, comm_, stream));
//   }
//   void nccl_recv(){
//     NCCLCHECK(ncclRecv((void*)recvbuff, size_t count, ncclFloat, peer, comm_, stream));
//   }
  void nccl_destory(){
    NCCLCHECK(ncclCommDestroy(comm_));
  }
private:
  ncclUniqueId id_;
  ncclComm_t comm_;
  int myRank_;
  int nRanks_;
};

__global__ void init_array(float* array, int length){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < length){
        array[idx] = 0;
    }
}

void init_gradient(float* x_gradient, int num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    init_array<<<blocknum, threadnum, 0, stream>>>(x_gradient, num_features);
    cudaCheckError();
}


void calc_dot(float* x_tmp, float* dr_a_norm_fp, int num_features, int num_samples, float* dots, cudaStream_t stream, cublasHandle_t handle){
    float alpha = 1.0;
    float beta = 0.0;
    int m = 1;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, num_samples, num_features, &alpha, x_tmp, m, dr_a_norm_fp, num_features, &beta, dots, m);
    cudaCheckError();
}


__global__ void calc_statistics_kernel(float* dot, float* dr_b, int num_samples, float* statisitcs){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_samples){
        statisitcs[idx] = dot[idx] - dr_b[idx];
    }
}

void calc_statistics(float* dot, float* dr_b, int num_samples, float* statisitcs, cudaStream_t stream){
    dim3 blocknum((num_samples - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    calc_statistics_kernel<<<blocknum, threadnum, 0, stream>>>(dot, dr_b, num_samples, statisitcs);
    cudaCheckError();
}

__global__ void calc_gradient_kernel(float* statisitcs, float* dr_a_norm_fp, int num_features, int num_samples, float stepSize_in_use, float* x_gradient){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features * num_samples){
        // if(statisitcs[idx / num_features]!=0){
        //     printf("%f\n", statisitcs[idx / num_features]);
        // }
        float delta = statisitcs[idx / num_features] * dr_a_norm_fp[idx] * stepSize_in_use;
        // if(stepSize_in_use ==0){
        //     printf("123\n");
        // }
        atomicAdd(x_gradient + (idx % num_features), delta);
    }
}


void calc_gradient(float* statisitcs, float* dr_a_norm_fp, int num_features, int num_samples, float stepSize_in_use, float* x_gradient, cudaStream_t stream, cublasHandle_t handle){
    // dim3 blocknum((num_samples * num_features - 1)/THREADNUM + 1, 1);
    // dim3 threadnum(THREADNUM, 1);
    // calc_gradient_kernel<<<blocknum, threadnum, 0, stream>>>(statisitcs, dr_a_norm_fp, num_features, num_samples, stepSize_in_use, x_gradient);
    // cudaCheckError();
    float alpha = stepSize_in_use;
    float beta = 1.0;
    int m = 1;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_features, m, num_samples, &alpha, dr_a_norm_fp, num_features, statisitcs, num_samples, &beta, x_gradient, num_features);
    cudaCheckError();
}

__global__ void update_xtmp_kernel(float* x_tmp, float* x_gradient, int num_features){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features){
        x_tmp[idx] -= x_gradient[idx];
    }
}

void update_xtmp(float* x_tmp, float* x_gradient, int num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    update_xtmp_kernel<<<blocknum, threadnum, 0, stream>>>(x_tmp, x_gradient, num_features);
    cudaCheckError();
}

__global__ void update_x_kernel(float* x, float* x_tmp, int num_features){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features){
        x[idx] = x_tmp[idx];
    }
}

void update_x(float* x, float* x_tmp, int num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    update_x_kernel<<<blocknum, threadnum, 0, stream>>>(x, x_tmp, num_features);
    cudaCheckError();
}

float calculate_loss(float* d_xtmp, int num_features, int num_samples, float* dr_b, float* dr_a_norm_fp) {
    //cout << "numSamples: "  << numSamples << endl;
   //cout << "numFeatures: " << numFeatures << endl;
   //numSamples  = 10;
   //cout << "For debugging: numSamples=" << numFeatures << endl;
   float* h_xtmp = (float*)malloc(num_features * sizeof(float));
   cudaMemcpy(h_xtmp, d_xtmp, num_features * sizeof(float), cudaMemcpyDeviceToHost);
   float loss = 0;
   for(int i = 0; i < num_samples; i++) {
       float dot = 0.0;
       for (int j = 0; j < num_features; j++) {
           dot += h_xtmp[j]*dr_a_norm_fp[i*num_features + j];
           //cout << "x["<< j <<"] =" << x[j] << "   a="<< a[i*numFeatures+ j];
       }
    //    std::cout<<"dot: "<<dot<<"\n";
       loss += (dot - dr_b[i])*(dot - dr_b[i]);
       //cout << "b[i]" << b[i] << endl;
       //cout << loss << endl;
    //    std::cout<<"loss "<<loss<<"\n";
   }

   loss /= (float)(2*num_samples);
   return loss;
}

void a_normalize(float* dr_a_norm_fp, float* dr_a, int dr_numFeatures, int dr_numSamples) 
{

	//uint32_t *data  = reinterpret_cast<uint32_t*>( myfpga->malloc(100)); 
	//dr_a_norm_fp = (float *)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a_norm_fp == NULL)
	{
		printf("Malloc dr_a_norm_fp failed in a_normalize\n");
		return;
	}

	//a_normalizedToMinus1_1 = toMinus1_1;
	// dr_a_norm   = (uint32_t *)malloc(dr_numSamples*dr_numFeatures*sizeof(uint32_t)); //to store the normalized result....
	// if (dr_a_norm == NULL)
	// {
	// 	printf("Malloc dr_a_norm failed in a_normalize\n");
	// 	return;
	// }

	float* dr_a_min    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the minimum value of features.....
	if (dr_a_min == NULL)
	{
		printf("Malloc dr_a_min failed in a_normalize\n");
		return;
	}

  float*	dr_a_max    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the miaximum value of features.....
	if (dr_a_max == NULL)
	{
		printf("Malloc dr_a_max failed in a_normalize\n");
		return;
	}

	//printf("dr_numFeatures = %d, dr_numSamples = %d, dr_numFeatures_algin = %d\n", dr_numFeatures, dr_numSamples, dr_numFeatures_algin);

	///Normalize the values in the whole column to the range {0, 1} or {-1, 1}/// 
	for (int j = 0; j < dr_numFeatures; j++) 
	{ // Don't normalize bias
		float amin = std::numeric_limits<float>::max();
		float amax = std::numeric_limits<float>::min();
		for (int i = 0; i < dr_numSamples; i++) 
		{
			float a_here = dr_a[i*dr_numFeatures + j];
			if (a_here > amax)
				amax = a_here;
			if (a_here < amin)
				amin = a_here;
		}
		dr_a_min[j]  = amin; //set to the global variable for pm
		dr_a_max[j]  = amax;

		float arange = amax - amin;
		if (arange > 0) 
		{
			for (int i = 0; i < dr_numSamples; i++) 
			{
				float tmp = ((dr_a[i*dr_numFeatures + j] - amin)/arange); //((dr_a[i*dr_numFeatures + j] - amin)/arange)*2.0-1.0;
			  	
			  	dr_a_norm_fp[i*dr_numFeatures + j] = tmp;
			  	// dr_a_norm[i*dr_numFeatures + j]    = (uint32_t) (tmp * 4294967295.0); //4294967296 = 2^32	
			}
		}
	}
}

// void load_libsvm_data(char* pathToFile, uint32_t numSamples, uint32_t numFeatures, uint32_t numBits, float* h_dr_a_norm_fp, float* h_dr_b) {
// 	std::cout << "Reading " << pathToFile << "\n";

// 	uint32_t dr_numSamples  = numSamples;
// 	uint32_t dr_numFeatures = numFeatures; // For the bias term

// 	//dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

// 	float* dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
// 	std::cout<<dr_numSamples*dr_numFeatures<<"\n";
// 	if (dr_a == NULL)
// 	{
// 		printf("Malloc dr_a failed in load_tsv_data\n");
// 		return;
// 	}
// 	std::cout << "dra " << "\n";
// 	//////initialization of the array//////
// 	for (long i = 0; i < dr_numSamples*dr_numFeatures; i++){
// 		dr_a[i] = 0.0;
// 	}

// 	std::cout << "draa " << "\n";
// 	float* dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
// 	if (dr_b == NULL)
// 	{
// 		printf("Malloc dr_b failed in load_tsv_data\n");
// 		return;
// 	}
// 	std::cout << "drb " << "\n";

// 	std::string line;
// 	std::ifstream f(pathToFile);

// 	int index = 0;
// 	if (f.is_open()) 
// 	{
// 		while( index < dr_numSamples ) 
// 		{
// 			// cout<<index<<endl;
// 			getline(f, line);
// 			int pos0 = 0;
// 			int pos1 = 0;
// 			int pos2 = 0;
// 			int column = 0;
// 			while ( pos2 != -1 ) //-1 (no bias...) //while ( column < dr_numFeatures ) 
// 			{
// 				if (pos2 == 0) 
// 				{
					
// 					pos2 = line.find(" ", pos1);
// 					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					
// 					dr_b[index] = temp;
// 					// cout << "dr_b: "  << temp << endl;
// 				}
// 				else 
// 				{
// 					pos0 = pos2;
// 					pos1 = line.find(":", pos1)+1;
// 					if(pos1==0){
// 						break;
// 					}
// 					// cout<<"pos:"<<pos1<<endl;
// 					pos2 = line.find(" ", pos1);
// 					column = stof(line.substr(pos0+1, pos1-pos0-1));
// 					if (pos2 == -1) 
// 					{
// 						pos2 = line.length()+1;
// 						dr_a[index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
// 					}
// 					else{
// 						dr_a[index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
// 					}
// 					// cout << "dr_a: "  << column << endl;
// 					//cout << "index*dr_numFeatures + column: "  << index*dr_numFeatures + column-1 << endl;
// 					//cout << "dr_a[index*dr_numFeatures + column]: "  << dr_a[index*dr_numFeatures + column-1] << endl;
// 				}
// 			}
// 			index++;
// 		}
// 		f.close();
// 	}
// 	else
//     std::cout << "Unable to open file " << pathToFile << "\n";

//     memcpy(h_dr_b, dr_b, dr_numSamples*sizeof(float));
//     // memcpy(h_dr_a_norm_fp, dr_a, dr_numSamples*dr_numFeatures*sizeof(float));
//     a_normalize(h_dr_a_norm_fp, dr_a, dr_numFeatures, dr_numSamples);
//     // int count = 0;

//     // for(int i = 0; i < dr_numSamples * dr_numFeatures; i++){
//     //     if(dr_a[i] != 0){
//     //         count++;
//     //         std::cout<<i/dr_numFeatures<<" "<<i%dr_numFeatures<<" ";
//     //         std::cout<<"dra: "<<dr_a[i]<<" count "<<count<<"\n";
//     //     } 
//     // }
//     // for(int i = 0; i < dr_numSamples; i++){
//     //     if(dr_b[i] != 0){
//     //         count++;
//     //         std::cout<<"drb: "<<dr_b[i]<<" count "<<count<<"\n";
//     //     } 
//     // }
// 	std::cout << "in libsvm, numSamples: "  << dr_numSamples << std::endl;
// 	std::cout << "in libsvm, numFeatures: " << dr_numFeatures << std::endl; 
// 	//std::cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << std::endl; 
// }

void load_libsvm_data(char* pathToFile, uint32_t numSamples, uint32_t numFeatures, uint32_t numBits, float* h_dr_a_norm_fp, float* h_dr_b) {
	std::cout << "Reading " << pathToFile << "\n";

	uint32_t dr_numSamples  = numSamples;
	uint32_t dr_numFeatures = numFeatures; // For the bias term

	//dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

	float* dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	std::cout<<dr_numSamples*dr_numFeatures<<"\n";
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	std::cout << "dra " << "\n";
	//////initialization of the array//////
	for (long i = 0; i < dr_numSamples*dr_numFeatures; i++){
		dr_a[i] = 0.0;
	}

	std::cout << "draa " << "\n";
	float* dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}
	std::cout << "drb " << "\n";

	std::string line;
	std::ifstream f(pathToFile);

	int index = 0;
	if (f.is_open()) 
	{
		while( index < dr_numSamples ) 
		{
			// cout<<index<<endl;
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;
			while ( pos2 != -1 ) //-1 (no bias...) //while ( column < dr_numFeatures ) 
			{
				if (pos2 == 0) 
				{
					
					pos2 = line.find(" ", pos1);
					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					
					dr_b[index] = temp;
					// cout << "dr_b: "  << temp << endl;
				}
				else 
				{
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					if(pos1==0){
						break;
					}
					// cout<<"pos:"<<pos1<<endl;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1));
					if (pos2 == -1) 
					{
						pos2 = line.length()+1;
						dr_a[index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else{
						dr_a[index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					// cout << "dr_a: "  << column << endl;
					//cout << "index*dr_numFeatures + column: "  << index*dr_numFeatures + column-1 << endl;
					//cout << "dr_a[index*dr_numFeatures + column]: "  << dr_a[index*dr_numFeatures + column-1] << endl;
				}
			}
			index++;
		}
		f.close();
	}
	else
    std::cout << "Unable to open file " << pathToFile << "\n";

    memcpy(h_dr_b, dr_b, dr_numSamples*sizeof(float));
    // memcpy(h_dr_a_norm_fp, dr_a, dr_numSamples*dr_numFeatures*sizeof(float));
    a_normalize(h_dr_a_norm_fp, dr_a, dr_numFeatures, dr_numSamples);
    // int count = 0;

    // for(int i = 0; i < dr_numSamples * dr_numFeatures; i++){
    //     if(dr_a[i] != 0){
    //         count++;
    //         std::cout<<i/dr_numFeatures<<" "<<i%dr_numFeatures<<" ";
    //         std::cout<<"dra: "<<dr_a[i]<<" count "<<count<<"\n";
    //     } 
    // }
    // for(int i = 0; i < dr_numSamples; i++){
    //     if(dr_b[i] != 0){
    //         count++;
    //         std::cout<<"drb: "<<dr_b[i]<<" count "<<count<<"\n";
    //     } 
    // }
	std::cout << "in libsvm, numSamples: "  << dr_numSamples << std::endl;
	std::cout << "in libsvm, numFeatures: " << dr_numFeatures << std::endl; 
	//std::cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << std::endl; 
}



int main(int argc, char** argv){
    uint32_t numberOfIterations = atoi(argv[1]);
    std::cout<<"epoch num: "<<numberOfIterations<<"\n";
    int stepSizeShifter = 12;
    float stepSize = 1.0/((float)(1<<stepSizeShifter));
    int mini_batch_size = atoi(argv[2]);
    std::cout<<"mini batch size: "<<mini_batch_size<<"\n";
    int dr_numSamples = atoi(argv[4]);//20000;
    std::cout<<"num samples: "<<dr_numSamples<<"\n";
    int dr_numFeatures = atoi(argv[5]);//21000;
    std::cout<<"num features: "<<dr_numFeatures<<"\n";
    float* h_dr_b = (float*)malloc(dr_numSamples * sizeof(float));
    float* h_dr_a_norm_fp = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float));
    load_libsvm_data((char*)argv[3], dr_numSamples, dr_numFeatures, 8, h_dr_a_norm_fp, h_dr_b);

    // std::ofstream fpe;
	//  fpe.open("../../../distribute_data/e.txt",ios::out);
	
    //  if(!fpe.is_open ())
    //     std::cout << "Open file failure" << std::endl;
    
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    int myRank; 
    int nRanks;
    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    ncclpack* pack = new ncclpack(myRank, nRanks);
    pack->nccl_init();
    std::cout<<"my rank: "<<myRank<<"\n";
    std::cout<<"nRanks: "<<nRanks<<"\n";
    int num_features_per_wk = dr_numFeatures;
    int num_samples_per_wk = mini_batch_size/nRanks;
	  float stepSize_in_use = stepSize/(float)mini_batch_size;
    int sample_offset = num_samples_per_wk * (myRank);
    //int micro_batch_size = 8;
    std::cout<<"samples per wk: "<<num_samples_per_wk<<"\n";
    std::cout<<"stepsize: "<<stepSize_in_use<<"\n";
    //std::cout<<"feature off: "<<feature_offset<<"\n";
    //dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
    
    float* x;
    cudaMalloc(&x, numberOfIterations * num_features_per_wk * sizeof(float));
    float* x_tmp;
    cudaMalloc(&x_tmp, num_features_per_wk * sizeof(float));
    cudaMemset(x_tmp, 0, num_features_per_wk * sizeof(float));
    float* x_gradient;
    cudaMalloc(&x_gradient, num_features_per_wk * sizeof(float));
    cudaMemset(x_gradient, 0, num_features_per_wk * sizeof(float));
    // float* x_gradient_tmp;
    // cudaMalloc(&x_gradient_tmp, num_features_per_wk * sizeof(float));
    // cudaMemset(x_gradient_tmp, num_features_per_wk * sizeof(float));

    // float* total_x_tmp;
    // cudaMalloc(&total_x_tmp, dr_numFeatures * sizeof(float));
    
    float* dr_a_norm_fp;
    cudaMalloc(&dr_a_norm_fp, dr_numSamples * num_features_per_wk * sizeof(float));
    // for(int i = 0; i < dr_numSamples/nRanks; i++){
    //     cudaMemcpy(dr_a_norm_fp + i * num_features_per_wk, h_dr_a_norm_fp + (i * nRanks + myRank) * dr_numFeatures, num_features_per_wk * sizeof(float), cudaMemcpyHostToDevice);
    // }
    cudaMemcpy(dr_a_norm_fp, h_dr_a_norm_fp, dr_numSamples * num_features_per_wk * sizeof(float), cudaMemcpyHostToDevice);
    float* dr_b;
    if(myRank == 0){
      float loss_value = calculate_loss(x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
      std::cout << "init_loss: "<< loss_value <<std::endl;
    }
    cudaMalloc(&dr_b, dr_numSamples * sizeof(float));
    cudaMemcpy(dr_b, h_dr_b, dr_numSamples * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreate(&stream2);
    cublasSetStream(handle, stream1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_elapsed;


    float* dot;
    cudaMalloc(&dot, num_samples_per_wk * sizeof(float));
    cudaMemset(dot, 0, num_samples_per_wk * sizeof(float));

    float* statisitcs;
    cudaMalloc(&statisitcs, num_samples_per_wk * sizeof(float));
    cudaMemset(statisitcs, 0, num_samples_per_wk * sizeof(float));


    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    if(nRanks == 11){
      for(int epoch = 0; epoch < numberOfIterations; epoch++) {
      //for one mini_batch...
        if(!graphCreated ){
          cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

          for (int i = 0; i < (dr_numSamples/mini_batch_size)*mini_batch_size; i += mini_batch_size) 
          {
                  init_gradient(x_gradient, num_features_per_wk, stream1);
      
                  calc_dot(x_tmp, dr_a_norm_fp + (i + sample_offset) * num_features_per_wk, num_features_per_wk, num_samples_per_wk, dot, stream1, handle);

                  calc_statistics(dot, dr_b + i + sample_offset, num_samples_per_wk, statisitcs, stream1);

                  calc_gradient(statisitcs, dr_a_norm_fp + (i + sample_offset) * num_features_per_wk, num_features_per_wk, num_samples_per_wk, stepSize_in_use, x_gradient, stream1, handle);
                  
                  pack->nccl_allreduce(x_gradient, x_gradient, num_features_per_wk, stream1);

                  update_xtmp(x_tmp, x_gradient, num_features_per_wk, stream1);
          }
    
          update_x(x + epoch * num_features_per_wk, x_tmp, num_features_per_wk, stream1);
          // std::cout << "epoch: "<< epoch << "dr_numFeatures: " << dr_numFeatures << "\n"; 
          // pack->nccl_allgather(x_tmp, total_x_tmp, num_features_per_wk, stream1);
          // cudaMemcpyAsync(total_x_tmp, x_tmp, num_features_per_wk * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
          // std::cout<<"nccl allgather done\n";
      
          cudaStreamEndCapture(stream1, &graph);
          cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
          cudaGraphNode_t* nodes = NULL;
          size_t numNodes = 0;
          cudaGraphGetNodes(graph, nodes, &numNodes);
          printf("Num of nodes in the graph created manually = %zu\n", numNodes);
          graphCreated=true;
          // if(myRank == 0){
          //   float loss_value = calculate_loss(x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
          //   std::cout << epoch << "_loss: "<< loss_value <<std::endl;
          // }
        }
        cudaEventRecord(start, stream1);
        cudaGraphLaunch(instance, stream1);
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(start);    //Waits for an event to complete.
        cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
        cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
        cudaStreamSynchronize(stream1);
        if(myRank == 0){
          float loss_value = calculate_loss(x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
          std::cout << epoch << "_loss: "<< loss_value <<std::endl;
          std::cout<<"time: "<<time_elapsed<<"\n";
        }
      }
    }else{
      //Iterate over each epoch...
      for(int epoch = 0; epoch < numberOfIterations; epoch++) 
      {   
        //for one mini_batch...
        cudaEventRecord(start, stream1);
        for (int i = 0; i < (dr_numSamples/mini_batch_size)*mini_batch_size; i += mini_batch_size) 
        {
                init_gradient(x_gradient, num_features_per_wk, stream1);
    
                calc_dot(x_tmp, dr_a_norm_fp + (i + sample_offset) * num_features_per_wk, num_features_per_wk, num_samples_per_wk, dot, stream1, handle);
                // cudaStreamSynchronize(stream1);
                // if(nRanks == 2 && myRank == 0)
                // std::cout<<"dot: "<<dot[0]<<"\n";

                // if(nRanks == 1){
                //   std::cout<<"dot: "<<dot[0]<<"\n";
                // }
                calc_statistics(dot, dr_b + i + sample_offset, num_samples_per_wk, statisitcs, stream1);
                // cudaStreamSynchronize(stream1);
                // if(nRanks == 2 && myRank == 0)
                // std::cout<<"sat: "<<statisitcs[0]<<"\n";
                // if(nRanks == 1){
                //   std::cout<<"stat: "<<statisitcs[0]<<"\n";
                // }
                calc_gradient(statisitcs, dr_a_norm_fp + (i + sample_offset) * num_features_per_wk, num_features_per_wk, num_samples_per_wk, stepSize_in_use, x_gradient, stream1, handle);
                // cudaStreamSynchronize(stream1);
                // if(myRank == 0){
                //   for(int j = 0; j < 200; j++){
                //     if(x_gradient[j]!=0){
                //       std::cout<<"gra: "<<x_gradient[j]<<"\n";
                //     }
                //   }
                // }


                pack->nccl_allreduce(x_gradient, x_gradient, num_features_per_wk, stream1);
                // cudaStreamSynchronize(stream1);

                // std::cout<<"all reduce\n";
                // if(myRank == 0){
                //   for(int j = 0; j < 200; j++){
                //     if(x_gradient[j]!=0){
                //       std::cout<<"gra: "<<x_gradient[j]<<"\n";
                //     }
                //   }
                // }
                update_xtmp(x_tmp, x_gradient, num_features_per_wk, stream1);

        }
  
        update_x(x + epoch * num_features_per_wk, x_tmp, num_features_per_wk, stream1);
          // cudaMemcpyAsync(total_x_tmp, x_tmp, num_features_per_wk * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
          // std::cout<<"nccl allgather done\n";
          cudaEventRecord(stop, stream1);
          cudaEventSynchronize(start);    //Waits for an event to complete.
          cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
          cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
          cudaStreamSynchronize(stream1);
          if(myRank == 0){
            float loss_value = calculate_loss(x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
            std::cout << epoch << "_loss: "<< loss_value <<std::endl;
            std::cout<<"time: "<<time_elapsed<<"\n";
          }
      }
    }

    

	//fpe.close();
    cublasDestroy(handle);
    pack->nccl_destory();
    MPICHECK(MPI_Finalize());
    printf("[MPI Rank %d] Success \n", myRank);
}

