#!/usr/bin/env bash

set -e

rm -rf djl
git clone https://github.com/awslabs/djl.git

cd djl/examples

# download the images for inference
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg
curl -O https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png

rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark

echo "Running inference Res18..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" > /tmp/benchmark/res18.log 2>&1
echo "Running inference Res50..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i 3dogs.jpg -r {'layers':'50','flavor':'v2'}" > /tmp/benchmark/res50.log 2>&1
echo "Running inference Res152..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i 3dogs.jpg -r {'layers':'152','flavor':'v1d'}" > /tmp/benchmark/res152.log 2>&1
echo "Running inference Res50Cifar10..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i 3dogs.jpg -r {'layers':'50','flavor':'v1','dataset':'cifar10'}" > /tmp/benchmark/res50_cifar10.log 2>&1
echo "Running inference Res50Cifar10 Imperative..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'}" > /tmp/benchmark/res50_cifar10_imp.log 2>&1
echo "Running inference SSD Resnet50..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i soccer.png -n SSD -r {'size':'512','backbone':'resnet50'}" > /tmp/benchmark/ssd_resnet50.log 2>&1
echo "Running inference SSD Vgg16..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 10 -i soccer.png -n SSD -r {'size':'512','backbone':'vgg16'}" > /tmp/benchmark/ssd_vgg16.log 2>&1

export MXNET_ENGINE_TYPE=NaiveEngine
echo "Running inference Res18 with NaiveEngine..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark -Dcollect-memory=true --args="-c 10 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/naive_res18.log 2>&1
echo "Running multithread inference Res18..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c 10 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/multithread_res18.log 2>&1
echo "Running multithread inference Res18 enableed threadsafe..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true -DMXNET_THREAD_SAFE_INFERENCE=true --args="-c 10 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/multithread_res18_threadsafe.log 2>&1
echo "Running multithread inference Res18 enableed threadsafe Imperative..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c 10 -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/multithread_res18_imp.log 2>&1
unset MXNET_ENGINE_TYPE

if ! nvidia-smi -L
then
    HW_TYPE=GPU
else
    HW_TYPE=CPU
fi

declare -a models=("res18" "res50" "res152" "res50_cifar10" "res50_cifar10_imp" "ssd_resnet50" "ssd_vgg16")
declare -a multithreading_models=("naive_res18" "multithread_res18" "multithread_res18_threadsafe" "multithread_res18_imp")

{
  printf "DJL Inference Result\n"
  printf "CPU/GPU: %s\n" "${HW_TYPE}"
  for model in "${models[@]}"; do
    printf "======================================\n"
    printf "%s Accurary: %s" "$model" "$(grep "inference P50:" /tmp/benchmark/"${model}".log | awk '{ print ($6, $9, $12, n) }')"
    printf "%s preprocess P50: %s" "$model" "$(grep "preprocess P50:" /tmp/benchmark/"${model}".log | awk '{ print ($6, $9, $12, n) }')"
    printf "%s postprocess P50: %s" "$model" "$(grep "preprocess P50:" /tmp/benchmark/"${model}".log | awk '{ print ($6, $9, $12, n) }')"
  done

  for multithreading_model in "${multithreading_models[@]}"; do
    printf "======================================\n"
    printf "%s inference P50: %s" "$multithreading_model" "$(grep "inference P50:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($6, $9, $12, n) }')"
    printf "%s preprocess P50: %s" "$multithreading_model" "$(grep "preprocess P50:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($6, $9, $12, n) }')"
    printf "%s postprocess P50: %s" "$multithreading_model" "$(grep "preprocess P50:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($6, $9, $12, n) }')"
    printf "%s heap: %s" "$multithreading_model" "$(grep "heap P90:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($NF, n) }')"
    printf "%s nonHeap: %s" "$multithreading_model" "$(grep "nonHeap P90:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($NF, n) }'))"
    printf "%s cpu: %s" "$multithreading_model" "$(grep "cpu P90:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($NF, n) }')"
    printf "%s rss: %s" "$multithreading_model" "$(grep "rss P90:" /tmp/benchmark/"${multithreading_model}".log | awk '{ print ($NF, n) }')"
  done

} >> /tmp/benchmark/report.txt
