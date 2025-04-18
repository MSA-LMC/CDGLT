import random
import numpy as np
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from M_loaddata import build_dataset, build_iterator
from M_model import Meteor
from M_train import train

task_names = ['sentiment category', 'sentiment degree', 'intention detection', 'offensiveness detection', 'metaphor occurrence']
task_classes_num = { # 它们分别是多少类别的分类任务：
    'sentiment category': 7,
    'sentiment degree': 3,
    'intention detection': 5,
    'offensiveness detection': 4,
    'metaphor occurrence': 2
}

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Make the behavior deterministic
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior in convolutions

init_seed(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, default=0, help='0: sentiment category, 1: sentiment degree, 2: intention detection, 3: offensiveness detection, 4: metaphor occurrence')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--batch-size', type=int, default=128, help='the size of a batch')
parser.add_argument('--learning-rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=200, help='total epochs for training')
parser.add_argument('--num-schedule-cycle', type=int, default=5, help='Number of cycles for learning rate scheduling within the total training epochs. Used in conjunction with CosineAnnealingLR, where T_max is set to total_epochs divided by this value.')
parser.add_argument('--require-improvement', type=int, default=20, help='control early stopping of the train') # 若超过`require_improvement`个epoch效果还没提升，则提前结束训练
opt = parser.parse_args()

DEVICE = torch.device('cpu')
if opt.device != 'cpu' and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{int(opt.device)}')

train_data, val_data, test_data = build_dataset(task_id=opt.task_id)

train_iter = build_iterator(train_data, batch_size=opt.batch_size, device=DEVICE)
val_iter = build_iterator(val_data, batch_size=opt.batch_size, device=DEVICE)
test_iter = build_iterator(test_data, batch_size=opt.batch_size, device=DEVICE)

print(f'Starting the task of `{task_names[opt.task_id]}`...')

model = Meteor(num_classes=task_classes_num[task_names[opt.task_id]]).to(DEVICE)
last_improve, bestValF1, timestamp = train(model, train_iter, val_iter, test_iter, args=opt)

# print(f'Training ends, the best marco F1 score in validation set is {bestValF1:.4} in the {last_improve + 1} Epoch\n')
print(f'So, model in epoch {last_improve + 1} should be chosen for test')
print(f"The timestamp of this experiment: {timestamp}")
