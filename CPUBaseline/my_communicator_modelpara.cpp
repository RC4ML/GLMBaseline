#include <stdio.h>
// #include "mpi.h"
#include "/usr/local/mpich-3.4.1/include/mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <limits> 

#include <vector>
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include "timer.h"

#define PARA_F 16
#define PARA_P 1
#define NUM_THREADS 20

int mini_batch_size;
int threads = 1;
int dr_numSamples;
int numberOfIterations;
pthread_barrier_t barrier;
int num_features_per_wk;
float stepSize_in_use;

float* x_tmp;
float** x_gradient;
float* dr_a_norm_fp;
float* dr_b;
float* dot;
float* dot_sum;
float* statisitcs;

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
      printf("Failed: MPI error %s:%d '%d'\n",        \
          __FILE__,__LINE__, e);   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  


void init_gradient(float** x_gradient, int num_features){

	// for (int k = 0; k < num_features; k++){ 
	// 	x_gradient[k] = 0.0;
	// }

	// memset(x_gradient, 0, num_features * sizeof(float));
//----------------------------------------------------------------
    for (int i = 0; i < num_features / PARA_F; i++){
        long long int bias_1 = 1l*i * PARA_F;//�?能溢�?
        __m512 tmp_para_d_grad = _mm512_set1_ps(0);

        for (int j = 0; j < threads; j++){
            _mm512_store_ps(x_gradient[j] + bias_1, tmp_para_d_grad);
        }
	}
}


void calc_dot(float* x_tmp, float* dr_a_norm_fp, int num_features, int num_samples, float* dots){

	// for(int i = 0; i < num_samples; i++){
	// 	dots[i] = 0;
	// 	for (int j = 0; j < num_features; j++){
	// 		dots[i] += x_tmp[j]*dr_a_norm_fp[i*num_features + j];
    //         // printf("x_tmp[j] = %f\t", x_tmp[j]);
    //         // printf("dr_a_norm_fp[i*num_features + j] = %f\t", dr_a_norm_fp[i*num_features + j]);
	// 	}	
    //     // printf("\n");
	// }

// #pragma omp parallel for
//     for (int td = 0; td < threads; td++) {
//         int num_samples_per_thread = num_samples / threads;
//         int bias_0 = td * num_samples_per_thread;

//         for(int i = 0; i < num_samples_per_thread; i++){
//             int bias_1 = (bias_0 + i) * num_features;
//             __m512 tmp_para_d_dots = _mm512_set1_ps(0);

// 		    for (int j = 0; j < num_features / PARA_F; j++){
//                 int bias_2 = j * PARA_F;
//                 __m512 tmp_para_d_dr = _mm512_load_ps((dr_a_norm_fp + bias_1 + bias_2));
//                 __m512 tmp_para_d_xtmp = _mm512_load_ps((x_tmp + bias_2));
//                 tmp_para_d_dots = _mm512_fmadd_ps(tmp_para_d_dr, tmp_para_d_xtmp, tmp_para_d_dots);
// 		    }	
//             dots[bias_0 + i] = _mm512_reduce_add_ps(tmp_para_d_dots);
//         }
//     }


	// for(int i = 0; i < num_samples; i++){
    //     // printf("num_features: %d\n", num_features);
    //     // printf("For test: \n");
    //     int bias_1 = i * num_features;
    //     __m512 tmp_para_d_dots = _mm512_set1_ps(0);
	// 	for (int j = 0; j < num_features / PARA_F; j++){
    //         int bias_2 = j * PARA_F;
    //         // printf("bias_2: %d\n", bias_2);
    //         __m512 tmp_para_d_dr = _mm512_load_ps(dr_a_norm_fp + bias_1 + bias_2);
    //         __m512 tmp_para_d_xtmp = _mm512_load_ps(x_tmp + bias_2);
    //         float *m2f = (float *)&tmp_para_d_xtmp;
    //         // for (int k = 0; k < PARA_F; k++){
	// 		//     printf("%f\t", m2f[k]);
	// 	    // }
    //         tmp_para_d_dots = _mm512_fmadd_ps(tmp_para_d_dr, tmp_para_d_xtmp, tmp_para_d_dots);
	// 	}	
    //     dots[i] = _mm512_reduce_add_ps(tmp_para_d_dots);
    //     // printf("\n");
	// }
//----------------------------------------------------------------
	for(int i = 0; i < num_samples; i++){
        long long int bias_1 = 1l*i * num_features;
        __m512 tmp_para_d_dots = _mm512_set1_ps(0);
		for (int j = 0; j < num_features / PARA_F; j++){
            int bias_2 = j * PARA_F;
            __m512 tmp_para_d_dr = _mm512_load_ps(dr_a_norm_fp + bias_1 + bias_2);
            __m512 tmp_para_d_xtmp = _mm512_load_ps(x_tmp + bias_2);
            float *m2f = (float *)&tmp_para_d_xtmp;
            tmp_para_d_dots = _mm512_fmadd_ps(tmp_para_d_dr, tmp_para_d_xtmp, tmp_para_d_dots);
		}	
        dots[i] = _mm512_reduce_add_ps(tmp_para_d_dots);
    }

}

void calc_statistics(float* dot, float* dr_b, int num_samples, float* statisitcs){

	for(int i = 0; i < num_samples; i++){
		statisitcs[i] = dot[i] - dr_b[i];
	}

// #pragma omp parallel for
//     for (int td = 0; td < threads; td++) {
//         int num_samples_per_thread = num_samples / threads;
//         int bias_0 = td * num_samples_per_thread;

// 	    for(int i = 0; i < num_samples_per_thread / PARA_P; i++){
//             int bias_1 = bias_0 + i * PARA_P;
//             __m512 tmp_para_d_dot = _mm512_load_ps((dot + bias_1));
//             __m512 tmp_para_d_dr = _mm512_load_ps((dr_b + bias_1));
//             __m512 sub_para_d = _mm512_sub_ps(tmp_para_d_dot, tmp_para_d_dr);
//             _mm512_store_ps(statisitcs + bias_1, sub_para_d);
// 	    }
//     }

	// for(int i = 0; i < num_samples / PARA_P; i++){
    //     int bias_1 = i * PARA_P;
    //     __m512 tmp_para_d_dot = _mm512_load_ps((dot + bias_1));
    //     __m512 tmp_para_d_dr = _mm512_load_ps((dr_b + bias_1));
    //     __m512 sub_para_d = _mm512_sub_ps(tmp_para_d_dot, tmp_para_d_dr);
    //     _mm512_store_ps(statisitcs + bias_1, sub_para_d);
	// }
}

void calc_gradient(float* statisitcs, float* dr_a_norm_fp, int num_features, int num_samples, float stepSize_in_use, float* x_gradient){

    // for(int i = 0; i < num_samples; i++){
	// 	for (int j = 0; j < num_features; j++){
	// 		x_gradient[j] += stepSize_in_use*statisitcs[i]*dr_a_norm_fp[i*num_features + j];
	// 	}
	// }

// #pragma omp parallel for
//     for (int td = 0; td < threads; td++) {
//         int num_samples_per_thread = num_samples / threads;
//         int bias_0 = td * num_samples_per_thread;

// 	    for(int i = 0; i < num_samples_per_thread; i++){
//             int bias_1 = (bias_0 + i) * num_features;
//             __m512 tmp_para_d_coeff = _mm512_set1_ps(stepSize_in_use * statisitcs[bias_0 + i]);

// 		    for (int j = 0; j < num_features / PARA_F; j++){
//                 int bias_2 = j * PARA_F;
//                 __m512 tmp_para_d_grad = _mm512_load_ps((x_gradient + bias_2));
//                 __m512 tmp_para_d_dr = _mm512_load_ps((dr_a_norm_fp + bias_1 + bias_2));
//                 tmp_para_d_grad = _mm512_fmadd_ps(tmp_para_d_coeff, tmp_para_d_dr, tmp_para_d_grad);
//                 _mm512_store_ps(x_gradient + bias_2, tmp_para_d_grad);
// 		    }
// 	    }
//     }
//-----------------------------------------------------------------------------
	for(int i = 0; i < num_samples; i++){
        long long int bias_1 = 1l*i * num_features;
        __m512 tmp_para_d_coeff = _mm512_set1_ps(stepSize_in_use * statisitcs[i]);
		for (int j = 0; j < num_features / PARA_F; j++){
            int bias_2 = j * PARA_F;
            __m512 tmp_para_d_grad = _mm512_load_ps((x_gradient + bias_2));
            __m512 tmp_para_d_dr = _mm512_load_ps((dr_a_norm_fp + bias_1 + bias_2));
            tmp_para_d_grad = _mm512_fmadd_ps(tmp_para_d_coeff, tmp_para_d_dr, tmp_para_d_grad);
            _mm512_store_ps(x_gradient + bias_2, tmp_para_d_grad);
		}
	}
}

void update_xtmp(float* x_tmp, float** x_gradient, int num_features){

	// for (int k = 0; k < num_features; k++){
	// 	x_tmp[k] -= x_gradient[k];
	// }

// #pragma omp parallel for
//     for (int td = 0; td < threads; td++) {
//         int num_features_per_thread = num_features / threads;
//         int bias_0 = td * num_features_per_thread;

// 	    for (int i = 0; i < num_features_per_thread / PARA_F; i++){
//             int bias_1 = bias_0 + i * PARA_F;
//             __m512 tmp_para_d_xtmp = _mm512_load_ps((x_tmp + bias_1));
//             __m512 tmp_para_d_grad = _mm512_load_ps((x_gradient + bias_1));
//             __m512 sub_para_d = _mm512_sub_ps(tmp_para_d_xtmp, tmp_para_d_grad);
//             _mm512_store_ps(x_tmp + bias_1, sub_para_d);
// 	    }    
//     }
//--------------------------------------------------------------
	for (int i = 0; i < num_features / PARA_F; i++){
        int bias_1 = i * PARA_F;
        __m512 tmp_para_d_xtmp = _mm512_load_ps((x_tmp + bias_1));
        __m512 tmp_para_d_grad_sum = _mm512_set1_ps(0);
        for (int j = 0; j < threads; j++){
            __m512 tmp_para_d_grad = _mm512_load_ps((x_gradient[j] + bias_1));
            tmp_para_d_grad_sum = _mm512_add_ps(tmp_para_d_grad, tmp_para_d_grad_sum);
        }
        __m512 sub_para_d = _mm512_sub_ps(tmp_para_d_xtmp, tmp_para_d_grad_sum);
        _mm512_store_ps(x_tmp + bias_1, sub_para_d);
	}
}

void update_x(float* x, float* x_tmp, int num_features){

	// for (int j = 0; j < num_features; j++) {
	// 	x[j] = x_tmp[j];
	// }

    // memcpy(x, x_tmp, num_features * sizeof(float));
//--------------------------------------------------------------------
    for (int i = 0; i < num_features / PARA_F; i++){
        int bias_1 = i * PARA_F;
        __m512 tmp_para_d_xtmp = _mm512_load_ps((x_tmp + bias_1));
        _mm512_store_ps(x + bias_1, tmp_para_d_xtmp);
	}
}

float calculate_loss(float* h_xtmp, int num_features, int num_samples, float* dr_b, float* dr_a_norm_fp) {
    //cout << "numSamples: "  << numSamples << endl;
   //cout << "numFeatures: " << numFeatures << endl;
   //numSamples  = 10;
   //cout << "For debugging: numSamples=" << numFeatures << endl;
   float loss = 0;
   for(long long int i = 0; i < num_samples; i++) {
       float dot = 0.0;
       for (int j = 0; j < num_features; j++) {
           dot += h_xtmp[j]*dr_a_norm_fp[1l * i*num_features + j];
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
			float a_here = dr_a[1l*i*dr_numFeatures + j];
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
				float tmp = ((dr_a[1l*i*dr_numFeatures + j] - amin)/arange); //((dr_a[i*dr_numFeatures + j] - amin)/arange)*2.0-1.0;
			  	
			  	dr_a_norm_fp[1l*i*dr_numFeatures + j] = tmp;
			  	// dr_a_norm[i*dr_numFeatures + j]    = (uint32_t) (tmp * 4294967295.0); //4294967296 = 2^32	
			}
		}
	}
}

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
	for (long i = 0; i < 1l*dr_numSamples*dr_numFeatures; i++){
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
						dr_a[1l*index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else{
						dr_a[1l*index*dr_numFeatures + column-1] = stof(line.substr(pos1, pos2-pos1), NULL);
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

void *model_task(void * id){
    int thread_id = *(int *)id;
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(thread_id,&mask);
	assert(!pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask));
    int mini_batch_size_per_thread = mini_batch_size / threads;
    int bias = mini_batch_size_per_thread * thread_id;
    int dr_numSamples_per_mini_batch_size = dr_numSamples / mini_batch_size;
    //printf("Im thread %d\n", thread_id);

    for(int epoch = 0; epoch < numberOfIterations; epoch++){   
        for (int i = 0; i < dr_numSamples; i += mini_batch_size){ 
            // //// printf("check point 0.0\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 0.1\n");
            // calc_dot(x_tmp, dr_a_norm_fp + (i + bias) * num_features_per_wk, num_features_per_wk, mini_batch_size_per_thread, dot + bias);
            // //// printf("check point 0.2\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 0.3\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 0.4\n");
            // calc_statistics(dot_sum + bias, dr_b + i + bias, mini_batch_size_per_thread, statisitcs + bias);
            // calc_gradient(statisitcs + bias, dr_a_norm_fp + (i + bias) * num_features_per_wk, num_features_per_wk, mini_batch_size_per_thread, stepSize_in_use, x_gradient);
            // //// printf("check point 0.5\n");
            // pthread_barrier_wait(&barrier);    
            // printf("check point 0.6\n");	
            // printf("check point 0.1\n");
            pthread_barrier_wait(&barrier);
            calc_dot(x_tmp, dr_a_norm_fp + 1l*(i + bias) * num_features_per_wk, num_features_per_wk, mini_batch_size_per_thread, dot + bias);
            // printf("check point 0.2\n");
            pthread_barrier_wait(&barrier);
            // printf("check point 0.3\n");
            pthread_barrier_wait(&barrier);
            // printf("check point 0.4\n");
            calc_statistics(dot_sum + bias, dr_b + i + bias, mini_batch_size_per_thread, statisitcs + bias);
            // pthread_barrier_wait(&barrier);
            // printf("check point 0.5\n");
            calc_gradient(statisitcs + bias, dr_a_norm_fp + 1l*(i + bias) * num_features_per_wk, num_features_per_wk, mini_batch_size_per_thread, stepSize_in_use, x_gradient[thread_id]); //x_gradient要改
            // printf("check point 0.6\n");
            pthread_barrier_wait(&barrier);
        }
    }
}

int main(int argc, char** argv){
    numberOfIterations = atoi(argv[1]);
    //std::cout<<"epoch num: "<<numberOfIterations<<"\n";
    //int stepSizeShifter = 12;
    int stepSizeShifter = 1;
    float stepSize = 1.0/((float)(1<<stepSizeShifter)); //�?一种写�?
    //float stepSize = 1.0;
    mini_batch_size = atoi(argv[2]);
    //std::cout<<"mini batch size: "<<mini_batch_size<<"\n";
    dr_numSamples = atoi(argv[4]);//20000;
    //std::cout<<"num samples: "<<dr_numSamples<<"\n";
    dr_numSamples = dr_numSamples / (PARA_P * threads) * (PARA_P * threads);
    //std::cout<<"num samples in pro: "<<dr_numSamples<<"\n";
    int dr_numFeatures = atoi(argv[5]);//21000;
    //std::cout<<"num features_input: "<<dr_numFeatures<<"\n";
    threads = atoi(argv[6]);//21000;
    //std::cout<<"threads: "<<threads<<"\n";

    dr_numSamples = dr_numSamples / mini_batch_size * mini_batch_size;
    //std::cout<<"num samples_tier: "<<dr_numSamples<<"\n";
    //std::cout<<"check point  1"<<"\n";
    int myRank; 
    int nRanks;
    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //std::cout<<"dr_numFeatures: "<<dr_numFeatures<<"\n";
    //std::cout<<"PARA_F * threads * nRanks: "<<PARA_F * threads * nRanks<<"\n";
    dr_numFeatures = (dr_numFeatures + PARA_F * threads * nRanks - 1) / (PARA_F * threads * nRanks) * (PARA_F * threads * nRanks);
    //std::cout<<"num features_working: "<<dr_numFeatures<<"\n";
    
    num_features_per_wk = dr_numFeatures / (nRanks);
	stepSize_in_use = stepSize/(float)mini_batch_size;
    // stepSize_in_use = 1;
    int feature_offset = num_features_per_wk * (myRank);
    //std::cout<<"features per wk: "<<num_features_per_wk<<"\n";
    //std::cout<<"stepsize: "<<stepSize_in_use<<"\n";
    //std::cout<<"feature off: "<<feature_offset<<"\n";
    //dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

    if (dr_numSamples % (PARA_P * threads) != 0){
        printf("[ERROR] dr_numSamples % (PARA_P * threads) != 0.\n");
        exit(2);
    }
    else if (dr_numSamples % mini_batch_size != 0){
        printf("[ERROR] dr_numSamples % mini_batch_size != 0.\n");
        exit(3);
    }  
    else if (mini_batch_size % threads != 0){
        printf("[ERROR] mini_batch_size % threads != 0.\n");
        exit(4);
    }  
    else if ((mini_batch_size / threads) % PARA_P != 0){
        printf("[ERROR] mini_batch_size / threads) % PARA_P != 0.\n");
        exit(5);
    }  

    float* h_dr_b = (float*)_mm_malloc(1l*dr_numSamples * sizeof(float), 64);
    memset(h_dr_b, 0, 1l*dr_numSamples * sizeof(float));

    float* h_dr_a_norm_fp = (float*)_mm_malloc(1l*dr_numSamples*dr_numFeatures*sizeof(float), 64); //暂时不�?�了
    memset(h_dr_a_norm_fp, 0, 1l*dr_numSamples*dr_numFeatures*sizeof(float));

    // load_libsvm_data((char*)argv[3], dr_numSamples, dr_numFeatures, 8, h_dr_a_norm_fp, h_dr_b); //数据就不读了

    float* x = (float*)_mm_malloc(1l*numberOfIterations * num_features_per_wk * sizeof(float), 64);
    memset(x, 0, 1l*numberOfIterations * num_features_per_wk * sizeof(float));

    x_tmp = (float*)_mm_malloc(1l*num_features_per_wk * sizeof(float), 64);
    memset(x_tmp, 0, 1l*num_features_per_wk * sizeof(float));

    x_gradient = (float**)malloc(1l*threads * sizeof(float*));

    for(int i = 0; i < threads; i++){
        x_gradient[i] = (float*)_mm_malloc(1l*num_features_per_wk * sizeof(float), 64);
        memset(x_gradient[i], 0, 1l*num_features_per_wk * sizeof(float));
    }

    float* total_x_tmp = (float*)_mm_malloc(1l*dr_numFeatures * sizeof(float), 64);
	memset(total_x_tmp, 0, dr_numFeatures * sizeof(float));

    dr_a_norm_fp = (float*)_mm_malloc(1l*dr_numSamples * num_features_per_wk * sizeof(float), 64);
    memset(dr_a_norm_fp, 0, 1l*dr_numSamples * num_features_per_wk * sizeof(float));

    for(int i = 0; i < dr_numSamples; i++){//暂时不�?�了
        memcpy(dr_a_norm_fp + 1l*i * num_features_per_wk, h_dr_a_norm_fp + 1l*i * dr_numFeatures + feature_offset, num_features_per_wk * sizeof(float));
    }

    dr_b = (float*)_mm_malloc(1l*dr_numSamples * sizeof(float), 64);
    memcpy(dr_b, h_dr_b, 1l*dr_numSamples * sizeof(float));
    // float loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp); 
    // std::cout << "init_loss: "<< loss_value <<std::endl;

    dot = (float*)_mm_malloc(1l*mini_batch_size * sizeof(float), 64);
    memset(dot, 0, 1l*mini_batch_size * sizeof(float));

	dot_sum = (float*)_mm_malloc(1l*mini_batch_size * sizeof(float), 64);
 	memset(dot_sum, 0, 1l*mini_batch_size * sizeof(float));

    statisitcs = (float*)_mm_malloc(1l*mini_batch_size * sizeof(float), 64);
    memset(statisitcs, 0, 1l*mini_batch_size * sizeof(float));

    // struct timespec start_timer,end_timer;
    // clock_gettime(CLOCK_MONOTONIC,&start_timer);

    CUtilTimer timer;
    CUtilTimer timer_allreduce;
    double accu_runtime = 0.0;

    CUtilTimer timer_part;

    // double accu_runtime_allreduce = 0.0;
    pthread_barrier_init(&barrier, NULL, threads + 1);
    pthread_t Thread[NUM_THREADS];
    int tmp[NUM_THREADS];
	for(int i = 0; i < threads; i++){
		tmp[i] = i;
		pthread_create(&Thread[i], NULL, model_task, &tmp[i]);
	}

    for(int epoch = 0; epoch < numberOfIterations; epoch++){  
        timer.start();
        for (int i = 0; i < dr_numSamples; i += mini_batch_size){
            // init_gradient(x_gradient, num_features_per_wk);
            // //// printf("check point 1.0\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 1.1\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 1.2\n");
            // MPICHECK(MPI_Allreduce(dot, dot_sum, mini_batch_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            // //// printf("check point 1.3\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 1.4\n");
            // pthread_barrier_wait(&barrier);
            // //// printf("check point 1.5\n");
            // update_xtmp(x_tmp, x_gradient, num_features_per_wk);
            //// printf("check point 1.6\n");
            // printf("check point 0.0\n");
            pthread_barrier_wait(&barrier); // 每两个barrier之间计时，batch 16，这里有4�?
            init_gradient(x_gradient, num_features_per_wk);
            // printf("check point 1.0\n");
            pthread_barrier_wait(&barrier);
            // printf("check point 2.0\n");
            
            // MPICHECK(MPI_Allreduce(dot, dot_sum, mini_batch_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            // timer_allreduce.start();
            MPICHECK(MPI_Allreduce(dot, dot_sum, mini_batch_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

            // printf("check point 3.0\n");
            pthread_barrier_wait(&barrier);
            // printf("check point 4.0\n");
            // pthread_barrier_wait(&barrier);
            // timer_allreduce.stop();

            // accu_runtime_allreduce = timer_allreduce.get_time();
            // std::cout <<" once_allreduce_time: "<< accu_runtime_allreduce <<std::endl;

            pthread_barrier_wait(&barrier);
            // printf("check point 5.0\n");
            update_xtmp(x_tmp, x_gradient, num_features_per_wk);
            // printf("check point 6.0\n");
        }
        //-----------计时
        update_x(x + epoch * num_features_per_wk, x_tmp, num_features_per_wk);
        MPICHECK(MPI_Allgather(x_tmp, num_features_per_wk, MPI_FLOAT, total_x_tmp, num_features_per_wk, MPI_FLOAT, MPI_COMM_WORLD));
        //------------

        timer.stop();
        double accu_runtime_per_epoch = timer.get_time();
        accu_runtime += accu_runtime_per_epoch;

        // if(myRank == 0){
        //     float loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
        //     std::cout << epoch << "_loss: "<< loss_value<<std::endl;
        //     // std::cout<<"time: "<<time_elapsed<<"\n";
        // }
        if(myRank == 0){
            std::cout<<"epoch: "<< epoch <<", time_per_epoch: "<<accu_runtime_per_epoch<<"\n";
        }
    }
    // clock_gettime(CLOCK_MONOTONIC,&end_timer);

    //double	m_end_time = ((double) end_timer.tv_sec + (double) end_timer.tv_nsec/1000000000.0) - ((double) start_timer.tv_sec + (double) start_timer.tv_nsec/1000000000.0);

        // if(myRank == 0){
        //     float loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
        //     std::cout << "_loss: "<< loss_value <<" time—�?1: "<< accu_runtime << " time—�?2: " << m_end_time <<std::endl;
        //     // std::cout<<"time: "<<time_elapsed<<"\n";
        // }

    // loss_value = calculate_loss(total_x_tmp, dr_numFeatures, dr_numSamples, h_dr_b, h_dr_a_norm_fp);
    // std::cout << "_loss: "<< loss_value <<" total_allreduce_time: "<< accu_runtime_allreduce <<std::endl;
    //std::cout<<"check point  5"<<"\n";
    if(myRank == 0){
        std::cout << (char*)argv[3] << " worker: " << nRanks << " batch: " << mini_batch_size << " total_time: "<< accu_runtime << " total_epoch: " << numberOfIterations << " time_per_epoch: " << accu_runtime / numberOfIterations <<std::endl;
    }
    MPICHECK(MPI_Finalize());
    //printf("[MPI Rank %d] Success \n", myRank);

}
