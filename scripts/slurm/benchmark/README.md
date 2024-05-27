# Minerva 1B GPU benchmark

1 GPU 1 Node

```bash
bash /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/base_scripts/train.sh \
    -c /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/benchmark/benchmark.yaml \
    -g 1 \
    -n 1 \
    -j benchmark-1gpu-1node \
    -x \
    -t 1:00:00
```

1 GPU 2 Node Full Shard

```bash
bash /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/base_scripts/train.sh \
    -c /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/benchmark/benchmark.yaml \
    -g 1 \
    -n 2 \
    -j benchmark/benchmark-1gpu-2node \
    -x \
    -t 1:00:00
```

4 GPU 1 Node Hybrid Shard

```bash
bash /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/base_scripts/train.sh \
    -c /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/benchmark/benchmark.yaml \
    -g 4 \
    -n 1 \
    -j benchmark/benchmark-4gpu-1node-withoutenvs \
    -x \
    -t 1:00:00
```

4 GPU 2 Node Hybrid Shard

```bash
bash /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/base_scripts/train.sh \
    -c /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/benchmark/benchmark.yaml \
    -g 4 \
    -n 2 \
    -j benchmark/benchmark-4gpu-2node-withoutenvs \
    -x \
    -t 1:00:00
```
