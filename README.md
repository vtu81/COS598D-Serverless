# Serverless Computing
### Assignment 3 for COS598D: System and Machine Learning

**Tinghao Xie (tx0973)**

<!-- In this assignment, you are required to implement and evaluate batched matrix multiplications as serverless calls. You are going to use [Submitit](https://github.com/facebookincubator/submitit) as the serverless framework. We provide an example of a serverless call performing an add operation in `example.py`. You are going to implement batched matrix multiplications in `batchMM.py` as serverless calls on a single CPU, multiple CPUs, and GPU.

## Getting Started
 - Clone this repo
 - Install [Submitit](https://github.com/facebookincubator/submitit) on a server with [Slurm](https://slurm.schedmd.com/quickstart.html) (e.g. [Adroit](https://researchcomputing.princeton.edu/systems/adroit) Server, or you can install Slurm on your server.) 
 - Install [Pytorch](https://pytorch.org/)

## How to Run 
Run `python example.py` for an serverless call example performing an add operation with Submitit.
You will need to fill in the `batchMM.py` file and run it using the command `python batchMM.py`. -->


## Environment

The experiments are run on a lab server with:
- **CPUs**: 4 * Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz (8 cores each)
- **GPUs**: 8 * Tesla P100-PCIE-16GB

## Tasks

### 1. Implement a Batched Matrix Multiplication Serverless Call

Implement a batched matrix multiplication serverless function (`bmm_single`) on a single CPU. 
The serverless call takes batched matrices *A*, and batched matrices *B* as input, computes batch matrix multiplication on *A* and *B* for 1,000 times on a single CPU, and returns batched output matrices *C*.
*A*, *B*, and *C* are Pytorch tensors of sizes `[bs, M, N]`, `[bs, N, K]`, and `[bs, M, K]` correspondingly. 

Implement the function `call_singleCPU` to make the serverless function call with randomnly initialized input matrices and verify the result correctness with the matrix parameters: `bs = 5, 10, 15` and `M = N = K = 100, 500, 1000`.


### 2. Compare the performance of serverless call with bare metal time

Measure the time for your serverless call and its bare metal time, i.e., the actual time it takes to run on the CPU server.
For each case in the following table, run your serverless call 5 times and report the times for all the 5 runs.

***Performance of Bare Metal v.s. Serverless Call of Batched Matrix Multiplication (Single CPU)***

|   [bs, M, N, K]  |   Bare Metal Time (s) |   Serverless Call Time (s) |  
|----------------|----------------|-------------|
| [5, 1000, 1000, 1000] | 152.5, 152.4, 150.1, 149.8, 150.0 |  155.2, 155.1, 152.1, 152.1, 153.1  |
| [10, 1000, 1000, 1000] | 270.3, 269.3, 270.1, 269.7, 269.7 |  272.5, 271.5, 272.5, 271.5, 272.5  |
| [15, 1000, 1000, 1000] | 419.8, 420.8, 421.6, 420.0, 420.0  | 423.0, 422.9, 425.0, 422.9, 422.9 |

Discuss the following
- Difference between the bare metal time and serverless call time and the reason.
    
    Answer: As shown, the serverless call time is always longer than the actual bare metal time by ~3s (overhead). The difference could be possibly and mainly caused by Slurm scheduling the job submitted by *submitit* -- i.e. *submitit* first needs to pickle and save the function and arguments to disk, *Slurm* needs a few seconds to put the received job in the queue, decide which devices to place the job, and then officially launch the job.
    
- Time variance across different runs and the reason.

    Answer: The time variance is rather small, compared to the elapsed serverless call time. Some potential explanations could be: 1) the jobs are scheduled on different devices, where they could have slightly different performance; 2) the temparature of the CPU has changed, leading to tiny difference of performance even on the same CPU; 3) lower level randomness (e.g. cache hit and miss) 


### 3. Parallelize Batch Matrix Multiplication Serverless Call on multiple CPUs

Implement a serverless call `mm_single` performing one matrix multiplication for 1,000 times on a single CPU.

Implement function `call_multiCPU` to parallelize the batched matrix multiplication by making one serverless call (`mm_single`) for each matrix multiplication within the batch, i.e., making `bs` serverless calls in parallel for batch size `bs`, and verify the result correctness.

For each case in the following table, run your parallelized serverless calls 5 times and report the median time.

***Performance of Serverless Batched Matrix Multiplication (Single CPU v.s. Multiple CPUs)***

|   `[bs, M, N, K]`  |   Single CPU (s) |   Multiple (`bs`) CPUs (s) |  
|----------------|----------------|-------------|
| [5, 100, 100, 100] | 2.2 |  4.2  |
| [10, 100, 100, 100] | 3.2 |  5.3 |
| [15, 100, 100, 100] | 3.2 |  5.6  |
| [5, 500, 500, 500] | 20.5 |  8.5  |
| [10, 500, 500, 500] | 40.7 |  10.8  |
| [15, 500, 500, 500] | 66.8 |  12.0  |
| [5, 1000, 1000, 1000] | 153.1 |  36.8  |
| [10, 1000, 1000, 1000] | 272.5 |  39.3  |
| [15, 1000, 1000, 1000] | 422.9 |  41.9  |

Discuss the performance difference between the single-CPU and the multi-CPU serverless call.
    
Answer: For matrix multiplication of larger scales (M/N/K ≥ 500), the multi-CPU serverless calls are significantly faster than the single-CPU calls. This benefit is obviously the benefit of parallelized computation -- `bs` CPU computes a smaller task at the same time. Yet, the serverless call time is not exactly `bs` times faster, compared to the single-CPU scenarios. This is mainly caused by the constant overhead of each job (as discussed in in the last section). In addition, the jobs are submitted and scheduled sequentially, which might further increase the overheads (e.g. pickling and saving the function and arguments to disk are done sequentially; and after all jobs finish, the main program needs to aggregate their results from distributed disck locations). And as a result, when computation workloads are small (M/N/K = 100), the overhead would dominate the serverless call time -- the multi-CPU call times are actually longer than the single-CPU call times.

### 4. Batch Matrix Multiplication Serverless Call on GPU

Implement the batched multiplication serverless call (`bmm_gpu`) for 1,000 times on GPU. 

Implement function `call_GPU` to make the serverless function call and verify its correctness.

For each case in the following table, run your serverless call 5 times and report the median time.

***Performance of Serverless Batched Matrix Multiplication (Single CPU v.s. Multiple CPUs v.s. GPU)***

|   `[bs, M, N, K]`  |  Single CPU (s) |   Multiple (`bs`) CPUs (s) |  GPU (s) |
|----------------|----------------|-------------|-------------|
| [5, 100, 100, 100] | 2.2 |  4.2  | 4.3 |
| [10, 100, 100, 100] | 3.2 |  5.3 | 4.2 |
| [15, 100, 100, 100] | 3.2 |  5.6  | 4.2 |
| [5, 500, 500, 500] | 20.5 |  8.5  | 4.3 |
| [10, 500, 500, 500] | 40.7 |  10.8  | 4.3 |
| [15, 500, 500, 500] | 66.8 |  12.0  | 4.3 |
| [5, 1000, 1000, 1000] | 153.1 |  36.8  | 5.4 |
| [10, 1000, 1000, 1000] | 272.5 |  39.3  | 6.5 |
| [15, 1000, 1000, 1000] | 422.9 |  41.9  | 7.5 |

Discuss the following
- Performance comparison of the three cases.
    
    Answer: In most cases, (single-)GPU serverless call is the fastest. While in the smallest case (bs = 5 and M/N/K = 100), single-CPU finishes the task first. As discussed earlier, this is caused by the constant overhead of serverless call (pickling the function and arguments, scheduling jobs, loading from disks, etc.) which could get enhanced in multi-job scenarios. When GPUs are introduced, some other overheads are introduced -- e.g. Slurm scheduling GPU job taking more time, copying data from main memory to GPU device memory, and so on. These additional overheads makes GPU the worst choice in the smallest case (4.3s > 4.2s > 2.2s).

- Pros and Cons for the three design choices: single CPU, multiple CPUs, and GPU.
    
    Answer:
    - Single CPU suffers from the least overhead (only have to pickling the function and arguments once, and Slurm only needs to schedule one job using one CPU core). Nevertheless, the computation power of a single CPU is the weakest, and thus medium/large tasks could take significantly more times, compared to the other two design choices.
    - Multiple CPUs allow us to perform matrix multiplication parallelly, speeding up the bare metal time of computation. But the multi-job fashion would also introduce multiple (and possibly larger) overhead, leading to actual throughput loss. When the workload is small, multi-CPU serverless call may be worse than single-CPU serverless call w.r.t. performance in time.
    - GPU is the most powerful device for matrix multiplication among the three design choices, thus the speedup of computation would be the best. Whilst, GPU introduces the additional overhead (e.g. copying data from host to device), and therefore is not suitable for small-scale computation.

- When serverless computing can be beneficial?
    
    Answer: Serverless computing is overall easy to use, where the user does not need to care about resource allocation and job submitting details. Yet, to benefit from serverless computing (e.g. the user may want to finish the computing quicker), either the user or the serverless computing system may need to place the job appropriately, considering the nature of the task (e.g. place small tasks on a single CPU to minimize the dominant overhead, and place large tasks on GPUs to make use of the huge amount of computation power while the overhead is relatively insignificant).

### 5. Batch Matrix Multiplication Serverless Call on multiple GPUs

Increase the number of batched matrix multiplication executions in batched multiplication serverless call on GPU (`bmm_gpu`) to 5,000 times. 

Implement function `call_multiGPU` to parallelize the batched matrix multiplication by spliting each batched matrix by half and making one serverless call (`bmm_gpu`) for each half batch, i.e., making 2 serverless calls in parallel, and verify the result's correctness.

For the case in the following table, run your serverless call 5 times and report the time for all the 5 runs.

***Performance of Serverless Batched Matrix Multiplication (Single GPU v.s. Multi GPUs)***

|   `[bs, M, N, K]`  | Bare Metal GPU (s)|  Single GPU (s) |   Multiple GPUs (s) | 
|----------------|----------------|-------------|-------------|
| [200, 1000, 1000, 1000] | 245.4, 243.2, 244.1, 243.0, 244.0 | 256.2, 253.7, 254.9, 252.1, 252.9 | 134.6, 132.7, 132.8, 131.7, 131.8 |

Discuss the performance and overhead for each of the three cases.

Answer: As shown, the overhead of a single GPU is ~10s (this does not include memory copy from host to device). This ~10s overhead includes pickling the matrix data and function to the disk and scheduling the GPU job via Slurm (similar to the Single-CPU case). Whilst, another implicit (but possibly significant; in this scale, ~1s) overhead, which is not included in this 10s but within the serverless call function, includes: copying the data from host memory to the GPU device memory (`.cuda()`); after computation, the results also need to be copied back to the host memory (`.cpu()`). Compared to the single-GPU scenario, the multi-GPU serverless call time is shorter (~52\%), benefiting from two GPU (each computing half of the workload) parallelly. Nevertheless, the scaling is non-linear (i.e. not approximate to the ideal 2x speedup). The potential reason behind this is the constant overhead (pickling the data, scheduling the job, etc.). Notice that once we subtract the 10s overhead from the mult-GPU call time, the bare metal time speedup would be around 2x.

<!-- ## What to be included in your submission

- A report includes your experiment settings, e.g., the server and environment you run your experiements on, the CPUs and GPUs you are using, etc., your results, discussions, and answers to all the questions in the previous section.
- Your `batchMM.py` file. -->
