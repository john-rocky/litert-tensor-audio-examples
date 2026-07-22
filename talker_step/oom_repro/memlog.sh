#!/system/bin/sh
cd /data/local/tmp/qwen3tts
OUT=memlog.txt
: > $OUT
i=0; PID=""
while [ -z "$PID" ] && [ $i -lt 50 ]; do PID=$(pidof talker_main); i=$((i+1)); sleep 0.2; done
echo "start pid=$PID $(cat /proc/uptime)" >> $OUT
while true; do
  MA=$(grep MemAvailable /proc/meminfo | tr -s ' ')
  if [ -n "$PID" ] && [ -d /proc/$PID ]; then RSS=$(grep VmRSS /proc/$PID/status | tr -s ' '); else RSS="proc-gone"; NP=$(pidof talker_main); [ -n "$NP" ] && PID=$NP; fi
  echo "$(cat /proc/uptime | cut -d' ' -f1) $MA | $RSS" >> $OUT
  sync
  sleep 1
done
