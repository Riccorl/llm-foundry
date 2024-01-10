import os
import submitit
import subprocess
import torch


def set_up_dist_env():
    # 1. RANK
    job_env = submitit.JobEnvironment()
    global_rank = job_env.global_rank

    # 2. LOCAL_RANK
    local_rank = job_env.local_rank

    # 3. LOCAL_WORLD_SIZE
    ngpus_per_node = torch.cuda.device_count()

    # 4. WORLD_SIZE
    world_size = int(os.getenv("SLURM_NNODES")) * ngpus_per_node

    # 5. NODE_RANK
    node_rank = int(os.getenv("SLURM_NODEID"))

    # 6. MASTER_ADDR
    cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]

    # 7. MASTER_PORT
    port = 54321

    # Set All the Necessary Environment Variables!
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(ngpus_per_node)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)
    os.environ["PYTHONUNBUFFERED"] = "1"


def submit_job():
    slurm_ngpus = 4
    slurm_nnodes = 2
    slurm_timeout = 1024
    workers = 10

    slurm_directory = "."  # "<Your Specified Directory>"
    executor = submitit.AutoExecutor(folder=slurm_directory)

    executor.update_parameters(
        mem_gb=128 * slurm_ngpus,
        gpus_per_node=slurm_ngpus,
        tasks_per_node=slurm_ngpus,
        cpus_per_task=workers,
        nodes=slurm_nnodes,
        timeout_min=slurm_timeout,
        slurm_partition="gpu",
        # see submitit github repo for details
    )

    job = executor.submit(train)


if __name__ == "__main__":
    submit_job()
