# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_next.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# # for i in {8..14}
# for i in {8..10}
# do
#     scp "/home/tt/experiment/cpu-training/communicator" tt@192.168.189.$i:"/home/tt/experiment/cpu-training/"
# done

# # np_arr=(1 2 4 8)
# # host_arr=("r1" "r1,r2" "r1,r2,r3,r4" "r1,r2,r3,r4,r5,r6,r7,r8")
# np_arr=(1 2 4)
# host_arr=("r1" "r1,r2" "r1,r2,r3,r4")

# # data_arr=("rcv1_train" "AMAZONFASION" "avazu")
# # size_arr=("20242 47236" "4096 332710" "2048 1000000")
# data_arr=("avazu")
# size_arr=("2048 1000000")
# # batch_arr=(16 64 256)
# batch_arr=(16)
# # thread_arr=(1 2 4 8 16)
# thread_arr=(4)

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 100 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} >> ./next_results_5/${data_arr[$j]}_pe_t${thread_arr[$l]}.txt
#             done
#         done
#     done
# done


# batch_arr2=(16 64 256 1024)
# for i in "${!batch_arr2[@]}"
# do
#     for j in "${!thread_arr[@]}"
#     do
#         mpirun -np 8 -host r1,r2,r3,r4,r5,r6,r7,r8 ./communicator 100 ${batch_arr2[$i]} unknowndataset 20480 1000000 ${thread_arr[$j]} >> ./next_results/e2_pe2_t${thread_arr[$j]}.txt
#     done
# done



#-----------------------------------------------------------------------

# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_final.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# # for i in {8..14}
# # do
# #     scp "/home/tt/experiment/cpu-training/communicator" tt@192.168.189.$i:"/home/tt/experiment/cpu-training/"
# # done

# # np_arr=(1 2 4 8)
# # host_arr=("r1" "r1,r2" "r1,r2,r3,r4" "r1,r2,r3,r4,r5,r6,r7,r8")
# np_arr=(1)
# host_arr=("r1")
# data_arr=("/home/tt/experiment/dataset/rcv1_train")
# size_arr=("20242 47236")
# batch_arr=(16)
# thread_arr=(1 2 4 8 16)

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 echo "mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 100 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} >> ./results/forcheck.txt"
#             done
#         done
#     done
# done


# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_next_2.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# for i in {8..14}
# do
#     scp "/home/tt/experiment/cpu-training/communicator" tt@192.168.189.$i:"/home/tt/experiment/cpu-training/"
# done

# np_arr=(8)
# host_arr=("r1,r2,r3,r4,r5,r6,r7,r8")

# # data_arr=("rcv1_train" "AMAZONFASION" "avazu")
# # size_arr=("20242 47236" "4096 332710" "2048 1000000")
# data_arr=("real-sim")
# size_arr=("72309 20958")
# batch_arr=(8)
# # batch_arr=(1024)
# thread_arr=(1)
# # thread_arr=(4)

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 1 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} >> ./next_results_8/b${batch_arr[$k]}_w${np_arr[$i]}.txt
#             done
#         done
#     done
# done


# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_datapara.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi




# np_arr=(1)
# host_arr=("r1")
# # data_arr=("rcv1_train" "AMAZONFASION" "avazu")
# # size_arr=("20242 47236" "4096 332710" "2048 1000000")
# data_arr=("/home/tt/experiment/dataset/rcv1_train")
# size_arr=("20242 47236")
# batch_arr=(8)
# thread_arr=(1,4,16)

# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_datapara.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 20 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} #>> ./next_results_8/b${batch_arr[$k]}_w${np_arr[$i]}.txt
#             done
#         done
#     done
# done

# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_dataparaCheck.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 20 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} #>> ./next_results_8/b${batch_arr[$k]}_w${np_arr[$i]}.txt
#             done
#         done
#     done
# done


np_arr=(4)
host_arr=("r1,r2,r3,r4")
data_arr=("amozon")
size_arr=("4096 332710")
batch_arr=(16 64 256 1024)
thread_arr=(1 2 4 8 16)

# g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_final.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

# for ((i=8; i<=10; i++))
# do
#     scp "/home/tt/experiment/cpu-training/communicator" tt@192.168.189.$i:"/home/tt/experiment/cpu-training/"
# done

# for i in "${!np_arr[@]}"
# do
#     for j in "${!data_arr[@]}"
#     do
#         for k in "${!batch_arr[@]}"
#         do
#             for l in "${!thread_arr[@]}"
#             do
#                 mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 50 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} >> ./modelpara_res/${data_arr[$j]}_w${np_arr[$i]}_b${batch_arr[$k]}_t${thread_arr[$l]}.txt
#             done
#         done
#     done
# done

g++ -o communicator -mavx512f -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512vl -mavx512dq -mavx512bw -pthread my_communicator_datapara.cpp ./timer.cpp -O3 -L /usr/local/mpich-3.4.1/lib/ -lmpi

for ((i=8; i<=10; i++))
do
    scp "/home/tt/experiment/cpu-training/communicator" tt@192.168.189.$i:"/home/tt/experiment/cpu-training/"
done

for i in "${!np_arr[@]}"
do
    for j in "${!data_arr[@]}"
    do
        for k in "${!batch_arr[@]}"
        do
            for l in "${!thread_arr[@]}"
            do
                mpirun -np ${np_arr[$i]} -host ${host_arr[$i]} ./communicator 50 ${batch_arr[$k]} ${data_arr[$j]} ${size_arr[$j]} ${thread_arr[$l]} >> ./datapara_res/${data_arr[$j]}_w${np_arr[$i]}_b${batch_arr[$k]}_t${thread_arr[$l]}.txt
            done
        done
    done
done
