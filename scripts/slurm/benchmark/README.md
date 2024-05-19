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
