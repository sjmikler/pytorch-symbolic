# Benchmarks

Fair benchmarking can be very difficult.

This is not a guide how to benchmark stuff correctly, but we'll share what we used to produce our charts.

We're using Linux with Intel CPU, some of the commands are different for AMD CPUs.

Before running any benchmarking script on thread `$THREAD`, we make sure to run the following:

```bash
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo 0 | tee /sys/kernel/randomize_va_space

echo performance > /sys/devices/system/cpu/cpu"$THREAD"/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu"$THREAD_NEXT"/cpufreq/scaling_governor

echo 1 > /sys/devices/system/cpu/cpu"$THREAD"/online
echo 0 > /sys/devices/system/cpu/cpu"$THREAD_NEXT"/online
```

# Requirement

NVIDIA's dllogger is required for logging. It can be downloaded from [https://github.com/NVIDIA/dllogger](https://github.com/NVIDIA/dllogger).

# Sources

* [https://llvm.org/docs/Benchmarking.html](https://llvm.org/docs/Benchmarking.html)

* [https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux)
