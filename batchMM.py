import submitit
import numpy as np
import torch
import time

def alloc_tensor(bs, M, N, K):
    tensor1 = torch.randn(bs, M, N)
    tensor2 = torch.randn(bs, N, K)
    return tensor1, tensor2

'''
Batched Matrix Multiplication
serverless call on CPU
'''
def bmm_single(a, b):
    assert len(a.shape) == 3 and len(b.shape) == 3
    start = time.time()
    # Run batched matrix multiplication on a and b for 1000 times
    for i in range(1000):
        # TODO: batched matrix multiplication on a and b
        result = torch.matmul(a, b)
    end = time.time()
    # Record the bare metal time
    rtime = end - start
    return result, rtime

'''
Matrix Multiplication
serverless call on CPU
'''
def mm_single(a, b):
    # Run one matrix multiplication on a and b for 1000 times
    assert len(a.shape) == 2 and len(b.shape) == 2
    for i in range(1000):
        # TODO: matrix multiplication on a and b
        result = torch.matmul(a, b)
    return result

'''
Batched Matrix Multiplication
serverless call on GPU
'''
def bmm_gpu(a, b):
    torch.cuda.synchronize()
    start = time.time()
    # TODO: copy a and b to GPU
    a, b = a.cuda(), b.cuda()
    # Run batched matrix multiplication on a and b for 1000 times
    for i in range(1000):
    # for i in range(5000):
        # TODO: batched matrix multiplication on GPU
        result = torch.matmul(a, b)
    # TODO: copy results back to CPU
    result = result.cpu()
    torch.cuda.synchronize()
    end = time.time()
    # Record the bare metal time
    rtime = end - start
    return result, rtime

'''
Make serverless call on a single CPU
'''
def call_singleCPU(tensor1, tensor2):
    assert len(tensor1.shape) == 3 and len(tensor2.shape) == 3
    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all")
    num_finished = 0
    # Record the serverless call time
    start = time.time()
    job = executor.submit(bmm_single, tensor1, tensor2)
    output, rtime = job.result()
    end = time.time()
    # TODO: verify result correctness
    true_result = torch.matmul(tensor1, tensor2)
    torch.testing.assert_close(true_result, output, rtol=1e-3, atol=1e-3)
    return end - start, rtime

'''
Make parallel serverless calls on multiple (bs) CPUs
'''
def call_multiCPU(tensor1, tensor2):
    # Submitit for multiple CPUs
    assert len(tensor1.shape) == 3 and len(tensor2.shape) == 3
    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all")
    num_finished = 0
    
    # Record the serverless call time
    start = time.time()
    
    # TODO: make bs parallel serverless calls to mm_single on bs CPUs
    bs = tensor1.shape[0]
    jobs = []
    for i in range(bs):
        job = executor.submit(mm_single, tensor1[i], tensor2[i])
        jobs.append(job)
    
    output = [job.result() for job in jobs]
    end = time.time()
    
    # TODO: verify result correctness
    output = torch.cat([o.unsqueeze(0) for o in output], dim=0)
    true_result = torch.matmul(tensor1, tensor2)
    torch.testing.assert_close(true_result, output, rtol=1e-3, atol=1e-3)
    
    return end - start

'''
Make serverless call on a single GPU
'''
def call_GPU(tensor1, tensor2):
    assert len(tensor1.shape) == 3 and len(tensor2.shape) == 3
    log_folder = "log_test/%j"
    
    # TODO: Define executer and specify parameters
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all", gpus_per_node=1)
    num_finished = 0
    
    # Record the serverless call time
    start = time.time()
    
    # TODO: make serverless call to bmm_gpu on GPU
    job = executor.submit(bmm_gpu, tensor1, tensor2)
    output, rtime = job.result()
    end = time.time()
    
    # TODO: verify result correctness
    tensor1, tensor2 = tensor1.cuda(), tensor2.cuda()
    true_result = torch.matmul(tensor1, tensor2).cpu()
    torch.testing.assert_close(true_result, output, rtol=1e-3, atol=1e-3)
    
    return end - start, rtime


'''
Make serverless call on multiple GPUs
'''
def call_multiGPU(tensor1, tensor2):
    assert len(tensor1.shape) == 3 and len(tensor2.shape) == 3
    log_folder = "log_test/%j"
    
    # TODO: Define executer and specify parameters
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all", gpus_per_node=1)
    num_finished = 0
    start = time.time()
    
    # TODO: make 2 parallel serverless calls to bmm_gpu on 2 GPUs, 
    # and increase the number of bmm execution in bmm_gpu to 5,000
    batch_size1 = tensor1.shape[0] // 2
    job1 = executor.submit(bmm_gpu, tensor1[:batch_size1], tensor2[:batch_size1])
    job2 = executor.submit(bmm_gpu, tensor1[batch_size1:], tensor2[batch_size1:])
    jobs = [job1, job2]
    output = [job.result()[0] for job in jobs]
    end = time.time()
    
    # TODO: verify result correctness
    output = torch.cat(output, dim=0)
    tensor1, tensor2 = tensor1.cuda(), tensor2.cuda()
    true_result = torch.matmul(tensor1, tensor2).cpu()
    torch.testing.assert_close(true_result, output, rtol=1e-3, atol=1e-3)
    
    return end - start


def main():
    # TODO: add more parameters and batch sizes
    paramList = [
        [100, 100, 100],
        [500, 500, 500],
        [1000, 1000, 1000]
    ]
    for bs in [5,10,15]:
    # for bs in [200]:
        for param in paramList:
            M = param[0]
            N = param[1]
            K = param[2]
            print("\nparam: [{}, {}, {}, {}]".format(bs, M, N, K))
            t1, t2 = alloc_tensor(bs, M, N, K)
            for r in range(5):
                print("[Single CPU]")
                time_singleCPU, rtime_singleCPU = call_singleCPU(t1, t2)
                print("Bare Metal Time", rtime_singleCPU)
                print("Serverless Call Time:", time_singleCPU)
                
                print("[Multi CPU]")
                time_multiCPU = call_multiCPU(t1, t2)
                print("Serverless Call Time:", time_multiCPU)
                
                print("[Single GPU]")
                time_GPU, rtime_GPU = call_GPU(t1, t2)
                print("Bare Metal Time", rtime_GPU)
                print("Serverless Call Time:", time_GPU)
                
                print("[Multi GPU]")
                time_multiGPU = call_multiGPU(t1, t2)
                print("Serverless Call Time:", time_multiGPU)

if __name__ == "__main__":
    main()