# /bin/bash

cd "$(dirname "$0")"/src
# ==============================

# nohup python -u main.py --task-id 3 --device 0 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 4 > task3a.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 0 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 4 > task3aa.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 0 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 4 > task3aaa.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 1 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 5 > task3b.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 1 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 5 > task3bb.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 1 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 5 > task3bbb.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 2 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 6 > task3c.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 2 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 6 > task3cc.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 2 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 6 > task3ccc.log 2>&1 &

# nohup python -u main.py --task-id 3 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 4 > task3d.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 3 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 4 > task3dd.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 5 > task3e.log 2>&1 &
nohup python -u main.py --task-id 3 --device 0 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 5 > task3ee.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 1 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 6 > task3f.log 2>&1 &
# nohup python -u main.py --task-id 3 --device 2 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 6 > task3ff.log 2>&1 &
