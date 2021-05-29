invoke profiler with (your java path): sudo nvvp -vm ~/java/jre1.8.0_291/bin/java
more info: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual
The profiler should be set to monitor ./tessla_cuda


for generating .log files: sudo nvprof --log-file profiler/<name>.log ./tessla_cuda

