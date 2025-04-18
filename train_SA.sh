# /bin/bash

cd "$(dirname "$0")"/src
# ==============================

# nohup python -u main.py --task-id 0 --device 0 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 4 > task0a.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 0 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 4 > task0aa.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 0 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 4 > task0aaa.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 1 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 5 > task0b.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 1 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 5 > task0bb.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 1 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 5 > task0bbb.log 2>&1 &
nohup python -u main.py --task-id 0 --device 2 --batch-size 128 --learning-rate 1e-4 --num-schedule-cycle 6 > task0c.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 2 --batch-size 128 --learning-rate 5e-4 --num-schedule-cycle 6 > task0cc.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 2 --batch-size 128 --learning-rate 1e-3 --num-schedule-cycle 6 > task0ccc.log 2>&1 &

# nohup python -u main.py --task-id 0 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 4 > task0d.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 3 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 4 > task0dd.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 3 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 5 > task0e.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 0 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 5 > task0ee.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 1 --batch-size 96 --learning-rate 5e-4 --num-schedule-cycle 6 > task0f.log 2>&1 &
# nohup python -u main.py --task-id 0 --device 2 --batch-size 96 --learning-rate 1e-4 --num-schedule-cycle 6 > task0ff.log 2>&1 &
