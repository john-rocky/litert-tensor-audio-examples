#!/system/bin/sh
cd /data/local/tmp/qwen3tts
rm -f memlog.txt talker_gpu_run.log
sh memlog.sh </dev/null >/dev/null 2>&1 &
sleep 1
LD_LIBRARY_PATH=. ./talker_main --layers=4 --prefill_len=64 --max_seq=1024 --steps=4 --warmup=0 --accelerator=gpu --gpu_serialization_dir=./clcache --gpu_cache_key=t4l --gpu_compile_threads=8 </dev/null > talker_gpu_run.log 2>&1
sync
