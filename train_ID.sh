# /bin/bash

cd "$(dirname "$0")"/src
# ==============================

nohup python -u main.py --task-id 2 --device 0 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 4 > task2a.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 0 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 4 > task2aa.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 0 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 4 > task2aaa.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 1 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 5 > task2b.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 1 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 5 > task2bb.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 1 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 5 > task2bbb.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 2 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 6 > task2c.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 2 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 6 > task2cc.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 2 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 6 > task2ccc.log 2>&1 &

# nohup python -u main.py --task-id 2 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 4 > task2d.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 3 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 4 > task2dd.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 5 > task2e.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 0 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 5 > task2ee.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 1 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 6 > task2f.log 2>&1 &
# nohup python -u main.py --task-id 2 --device 2 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 6 > task2ff.log 2>&1 &
