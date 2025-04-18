# /bin/bash

cd "$(dirname "$0")"/src
# ==============================

# nohup python -u main.py --task-id 4 --device 0 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 4 > task4a.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 0 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 4 > task4aa.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 0 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 4 > task4aaa.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 1 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 5 > task4b.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 1 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 5 > task4bb.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 1 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 5 > task4bbb.log 2>&1 &
nohup python -u main.py --task-id 4 --device 2 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 6 > task4c.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 2 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 6 > task4cc.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 2 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 6 > task4ccc.log 2>&1 &

# nohup python -u main.py --task-id 4 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 4 > task4d.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 3 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 4 > task4dd.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 5 > task4e.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 0 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 5 > task4ee.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 1 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 6 > task4f.log 2>&1 &
# nohup python -u main.py --task-id 4 --device 2 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 6 > task4ff.log 2>&1 &

