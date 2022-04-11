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
#define SEG 4
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

__global__ void init_array(float* array, long length){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < length){
        array[idx] = 0;
    }
}

void init_gradient(float* x_gradient, long num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    init_array<<<blocknum, threadnum, 0, stream>>>(x_gradient, num_features);
    cudaCheckError();
}


void calc_dot(float* x_tmp, float* dr_a_norm_fp, long num_features, long num_samples, float* dots, cudaStream_t stream, cublasHandle_t handle){
    float alpha = 1.0;
    float beta = 0.0;
    long m = 1;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, num_samples, num_features, &alpha, x_tmp, m, dr_a_norm_fp, num_features, &beta, dots, m);
    cudaCheckError();
}


__global__ void calc_statistics_kernel(float* dot, float* dr_b, long num_samples, float* statisitcs){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_samples){
        statisitcs[idx] = dot[idx] - dr_b[idx];
    }
}

void calc_statistics(float* dot, float* dr_b, long num_samples, float* statisitcs, cudaStream_t stream){
    dim3 blocknum((num_samples - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    calc_statistics_kernel<<<blocknum, threadnum, 0, stream>>>(dot, dr_b, num_samples, statisitcs);
    cudaCheckError();
}

__global__ void calc_gradient_kernel(float* statisitcs, float* dr_a_norm_fp, long num_features, long num_samples, float stepSize_in_use, float* x_gradient){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features * num_samples){
        // if(statisitcs[idx / num_features]!=0){
        //     prlongf("%f\n", statisitcs[idx / num_features]);
        // }
        float delta = statisitcs[idx / num_features] * dr_a_norm_fp[idx] * stepSize_in_use;
        // if(stepSize_in_use ==0){
        //     prlongf("123\n");
        // }
        atomicAdd(x_gradient + (idx % num_features), delta);
    }
}


void calc_gradient(float* statisitcs, float* dr_a_norm_fp, long num_features, long num_samples, float stepSize_in_use, float* x_gradient, cudaStream_t stream, cublasHandle_t handle){
    // dim3 blocknum((num_samples * num_features - 1)/THREADNUM + 1, 1);
    // dim3 threadnum(THREADNUM, 1);
    // calc_gradient_kernel<<<blocknum, threadnum, 0, stream>>>(statisitcs, dr_a_norm_fp, num_features, num_samples, stepSize_in_use, x_gradient);
    // cudaCheckError();
    float alpha = stepSize_in_use;
    float beta = 1.0;
    long m = 1;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_features, m, num_samples, &alpha, dr_a_norm_fp, num_features, statisitcs, num_samples, &beta, x_gradient, num_features);
    cudaCheckError();
}

__global__ void update_xtmp_kernel(float* x_tmp, float* x_gradient, long num_features){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features){
        x_tmp[idx] -= x_gradient[idx];
    }
}

void update_xtmp(float* x_tmp, float* x_gradient, long num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    update_xtmp_kernel<<<blocknum, threadnum, 0, stream>>>(x_tmp, x_gradient, num_features);
    cudaCheckError();
}

__global__ void update_x_kernel(float* x, float* x_tmp, long num_features){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_features){
        x[idx] = x_tmp[idx];
    }
}

void update_x(float* x, float* x_tmp, long num_features, cudaStream_t stream){
    dim3 blocknum((num_features - 1)/THREADNUM + 1, 1);
    dim3 threadnum(THREADNUM, 1);
    update_x_kernel<<<blocknum, threadnum, 0, stream>>>(x, x_tmp, num_features);
    cudaCheckError();
}

float calculate_loss(float* d_xtmp, long num_features, long num_samples, float* dr_b, float* dr_a_norm_fp) {
    //cout << "numSamples: "  << numSamples << endl;
   //cout << "numFeatures: " << numFeatures << endl;
   //numSamples  = 10;
   //cout << "For debugging: numSamples=" << numFeatures << endl;
   float* h_xtmp = (float*)malloc(num_features * sizeof(float));
   cudaMemcpy(h_xtmp, d_xtmp, num_features * sizeof(float), cudaMemcpyDeviceToHost);
   float loss = 0;
   for(long i = 0; i < num_samples; i++) {
       float dot = 0.0;
       for (long j = 0; j < num_features; j++) {
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

void a_normalize(float* dr_a_norm_fp, float* dr_a, long dr_numFeatures, long dr_numSamples) 
{

	//ulong32_t *data  = relongerpret_cast<ulong32_t*>( myfpga->malloc(100)); 
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
	for (long j = 0; j < dr_numFeatures; j++) 
	{ // Don't normalize bias
		float amin = std::numeric_limits<float>::max();
		float amax = std::numeric_limits<float>::min();
		for (long i = 0; i < dr_numSamples; i++) 
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
			for (long i = 0; i < dr_numSamples; i++) 
			{
				float tmp = ((dr_a[i*dr_numFeatures + j] - amin)/arange); //((dr_a[i*dr_numFeatures + j] - amin)/arange)*2.0-1.0;
			  	
			  	dr_a_norm_fp[i*dr_numFeatures + j] = tmp;
			  	// dr_a_norm[i*dr_numFeatures + j]    = (uint32_t) (tmp * 4294967295.0); //4294967296 = 2^32	
			}
		}
	}
}

void load_libsvm_data(char* pathToFile, long numSamples, long numFeatures, uint32_t numBits, float* h_dr_a_norm_fp, float* h_dr_b) {
	std::cout << "Reading " << pathToFile << "\n";

	long dr_numSamples  = numSamples;
	long dr_numFeatures = numFeatures; // For the bias term

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

	long index = 0;
	if (f.is_open()) 
	{
		while( index < dr_numSamples ) 
		{
			// cout<<index<<endl;
			getline(f, line);
			long pos0 = 0;
			long pos1 = 0;
			long pos2 = 0;
			long column = 0;
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
    uint32_t numberOfIterations = 20;//atoi(argv[1]);
    std::cout<<"epoch num: "<<numberOfIterations<<"\n";
    int stepSizeShifter = atoi(argv[5]);
    float stepSize = 1.0/((float)(1<<stepSizeShifter));
    long mini_batch_size = atoi(argv[4]);
    std::cout<<"mini batch size: "<<mini_batch_size<<"\n";
    long dr_numSamples = atoi(argv[2]);//20000;
    std::cout<<"num samples: "<<dr_numSamples<<"\n";
    long dr_numFeatures = atoi(argv[3]);//21000;
    std::cout<<"num features: "<<dr_numFeatures<<"\n";
    float* h_dr_b = (float*)malloc(dr_numSamples * sizeof(float));
    float* h_dr_a_norm_fp = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float));

    load_libsvm_data((char*)argv[1], dr_numSamples, dr_numFeatures, 8, h_dr_a_norm_fp, h_dr_b);
    std::ofstream destFile(argv[6],std::ios::out);
    if(!destFile) {
      std::cout << "error opening destination file." << std::endl;
      return 0;
  }
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

    long num_features_per_wk = dr_numFeatures / (nRanks);
	  float stepSize_in_use = stepSize/(float)mini_batch_size;
    long feature_offset = num_features_per_wk * (myRank);
    long micro_batch_size = mini_batch_size;
    std::cout<<"micro batch size: "<<micro_batch_size<<"\n";
    std::cout<<"features per wk: "<<num_features_per_wk<<"\n";
    std::cout<<"stepsize: "<<stepSize_in_use<<"\n";
    std::cout<<"feature off: "<<feature_offset<<"\n";
    //dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
    
    float* x;
    cudaMalloc(&x, numberOfIterations * num_features_per_wk * sizeof(float));
    float* x_tmp;
    cudaMalloc(&x_tmp, num_features_per_wk * sizeof(float));
    cudaMemset(x_tmp, 0, num_features_per_wk * sizeof(float));
    float* x_gradient;
    cudaMalloc(&x_gradient, num_features_per_wk * sizeof(float));
    cudaMemset(x_gradient, 0, num_features_per_wk * sizeof(float));
    
    float* total_x_tmp;
    cudaMalloc(&total_x_tmp, dr_numFeatures * sizeof(float));
    
    float** dr_a_norm_fp;
    cudaMallocManaged(&dr_a_norm_fp, SEG * sizeof(float*));
    for(int i = 0; i < SEG ; i++){
      float* new_fp;
      cudaMalloc(&new_fp, dr_numSamples * num_features_per_wk * sizeof(float) / SEG);
      dr_a_norm_fp[i] = new_fp;
    }
    cudaCheckError();
    // cudaMalloc(&dr_a_norm_fp, dr_numSamples * num_features_per_wk * sizeof(float));
    for(long i = 0; i < dr_numSamples; i++){
        cudaMemcpy( dr_a_norm_fp[i/(dr_numSamples/SEG)] + (i % (dr_numSamples/SEG)) * num_features_per_wk, h_dr_a_norm_fp + i * dr_numFeatures + feature_offset, num_features_per_wk * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaCheckError();
    float* dr_b;
    float loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
    std::cout << "init_loss: "<< loss_value <<std::endl;
    // float threshold = atof(argv[6]);
    // float loss_old = loss_value;

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
    float sum_time = 0;
    float* dot;
    cudaMalloc(&dot, micro_batch_size * sizeof(float));
    cudaMemset(dot, 0, micro_batch_size * sizeof(float));

    float* statisitcs;
    cudaMalloc(&statisitcs, micro_batch_size * sizeof(float));
    cudaMemset(statisitcs, 0, micro_batch_size * sizeof(float));


    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    long epoch = 0;
    if(nRanks == 1){
      for(epoch = 0; epoch < numberOfIterations; epoch++) {
      //for one mini_batch...
        if(!graphCreated ){
          cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

          for (long i = 0; i < (dr_numSamples/mini_batch_size)*mini_batch_size; i += mini_batch_size) 
          {
                  init_gradient(x_gradient, num_features_per_wk, stream1);
                  //std::cout<<"init gradient done\n";
                  for(long kk = 0; kk < mini_batch_size / micro_batch_size; kk += 1){//worker
                      calc_dot(x_tmp, dr_a_norm_fp[(i  + kk * micro_batch_size)/(dr_numSamples/SEG)] + ((i  + kk * micro_batch_size) % (dr_numSamples/SEG)) * num_features_per_wk, num_features_per_wk, micro_batch_size, dot, stream1, handle);

                      pack->nccl_allreduce(dot, dot, micro_batch_size, stream1);
    
                      calc_statistics(dot, dr_b + i  + kk * micro_batch_size, micro_batch_size, statisitcs, stream1);
    
                      calc_gradient(statisitcs, dr_a_norm_fp[(i  + kk * micro_batch_size)/(dr_numSamples/SEG)] + ((i  + kk * micro_batch_size) % (dr_numSamples/SEG)) * num_features_per_wk, num_features_per_wk, micro_batch_size, stepSize_in_use, x_gradient, stream1, handle);
                  }
                  update_xtmp(x_tmp, x_gradient, num_features_per_wk, stream1);
          }
    
          update_x(x + epoch * num_features_per_wk, x_tmp, num_features_per_wk, stream1);
          // std::cout << "epoch: "<< epoch << "dr_numFeatures: " << dr_numFeatures << "\n"; 
          pack->nccl_allgather(x_tmp, total_x_tmp, num_features_per_wk, stream1);
          // cudaMemcpyAsync(total_x_tmp, x_tmp, num_features_per_wk * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
          // std::cout<<"nccl allgather done\n";
      
          cudaStreamEndCapture(stream1, &graph);
          cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
          cudaGraphNode_t* nodes = NULL;
          size_t numNodes = 0;
          cudaGraphGetNodes(graph, nodes, &numNodes);
          printf("Num of nodes in the graph created manually = %zu\n", numNodes);
          graphCreated=true;
        }
        cudaEventRecord(start, stream1);
        cudaGraphLaunch(instance, stream1);
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(start);    //Waits for an event to complete.
        cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
        cudaEventElapsedTime(&time_elapsed,start,stop);    
        cudaStreamSynchronize(stream1);
        sum_time+=time_elapsed;
        // if(myRank == 0){
          loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
          // std::cout << epoch << "_loss: "<< loss_value <<std::endl;
          if(myRank == 0){
            destFile << sum_time  << " " <<loss_value <<"\n";
          }
          // if((loss_old - loss_value) < threshold){
          //   break;
          // }
          // loss_old = loss_value;
          // std::cout<<"time: "<<time_elapsed<<"\n";
        // }
      }
    }else{
      //Iterate over each epoch...
      for(epoch = 0; epoch < numberOfIterations; epoch++) 
      {   
        //for one mini_batch...
        cudaEventRecord(start, stream1);
          for (long i = 0; i < (dr_numSamples/mini_batch_size)*mini_batch_size; i += mini_batch_size) 
          {       
                  init_gradient(x_gradient, num_features_per_wk, stream1);
                  //std::cout<<"init gradient done\n";
                  for(long kk = 0; kk < mini_batch_size / micro_batch_size; kk += 1){//worker
                      calc_dot(x_tmp, dr_a_norm_fp[(i  + kk * micro_batch_size)/(dr_numSamples/SEG)] + ((i  + kk * micro_batch_size) % (dr_numSamples/SEG)) * num_features_per_wk, num_features_per_wk, micro_batch_size, dot, stream1, handle);
                      
                      pack->nccl_allreduce(dot, dot, micro_batch_size, stream1);
    
                      calc_statistics(dot, dr_b + i  + kk * micro_batch_size, micro_batch_size, statisitcs, stream1);

                      calc_gradient(statisitcs, dr_a_norm_fp[(i  + kk * micro_batch_size)/(dr_numSamples/SEG)] + ((i  + kk * micro_batch_size) % (dr_numSamples/SEG)) * num_features_per_wk, num_features_per_wk, micro_batch_size, stepSize_in_use, x_gradient, stream1, handle);

                  }
                  update_xtmp(x_tmp, x_gradient, num_features_per_wk, stream1);
                 
          }
    
          update_x(x + epoch * num_features_per_wk, x_tmp, num_features_per_wk, stream1);
          // std::cout << "epoch: "<< epoch << "dr_numFeatures: " << dr_numFeatures << "\n"; 
          pack->nccl_allgather(x_tmp, total_x_tmp, num_features_per_wk, stream1);
          // cudaMemcpyAsync(total_x_tmp, x_tmp, num_features_per_wk * sizeof(float), cudaMemcpyDeviceToDevice, stream1);
          // std::cout<<"nccl allgather done\n";
   
          cudaEventRecord(stop, stream1);
          cudaEventSynchronize(start);    //Waits for an event to complete.
          cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
          cudaEventElapsedTime(&time_elapsed,start,stop); 
          sum_time+=time_elapsed;
    
          loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
          if(myRank == 0){
            destFile << sum_time  << " " <<loss_value <<"\n";
          }
          // if((loss_old - loss_value) < threshold){
          //   break;
          // }
          // loss_old = loss_value;
  
      }
    }
    if(myRank == 0){
      std::cout << "batch_size: "<<mini_batch_size<<" shifter: "<<stepSizeShifter<<"\n";
      std::cout << epoch << "_loss: "<< loss_value <<"\n";
      std::cout <<"time: "<<sum_time<<"\n";
    }

    destFile.close();

	//fpe.close();
    cublasDestroy(handle);
    pack->nccl_destory();
    MPICHECK(MPI_Finalize());
    printf("[MPI Rank %d] Success \n", myRank);
}

