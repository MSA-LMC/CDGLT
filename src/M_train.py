import numpy as np
import torch
from sklearn import metrics
import time
from datetime import timedelta

class StateRecorder():
    def __init__(self, timestamp=""):
        super(StateRecorder, self)
        self.timestamp = timestamp
        self.loss_func = torch.nn.functional.cross_entropy
        self.bestValF1 = 0

stateRecorder = StateRecorder()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, train_iter, val_iter, test_iter, args):
    start_time = time.time()
    stateRecorder.timestamp = "".join(str(start_time).split('.'))
    print("\n==========")
    print(f"The timestamp of this experiment: {stateRecorder.timestamp}")
    stateRecorder.loss_func = model.calculate_loss
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_schedule_cycle = args.num_schedule_cycle
    require_improvement = args.require_improvement
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//num_schedule_cycle, eta_min=5e-6)
    last_improve = 0  # 记录上次验证集loss下降的epoch
    for epoch in range(num_epochs):
        print("==========\n")
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        total_loss = 0
        for i, (trains, labels) in enumerate(train_iter):
            image_features = trains[0]
            text_features = trains[1]
            prompt_input_ids = trains[2]
            prompt_attention_mask = trains[3]
            optimizer.zero_grad() # 清空梯度
            outputs = model(image_features, text_features, prompt_input_ids, prompt_attention_mask)  # 前向传播 获取输出
            loss = stateRecorder.loss_func(outputs, labels.long())  # 计算交叉熵损失
            total_loss += loss.item()
            loss.backward()  # 计算梯度
            optimizer.step() # 更新参数
        # 每一个epoch在验证集上计算相关指标
        train_loss = total_loss / len(train_iter) # 训练损失
        scheduler.step() # 学习率调度器以1个epoch为1步 # *****
        dev_acc, dev_macro_avg_f1, dev_loss = evaluate(model, val_iter) #计算此时模型在验证集上的损失和准确率
        time_dif = get_time_dif(start_time)
        msg = 'Train Loss: {0:>5.2},  Val Loss: {1:>5.2},  Val Acc: {2:>6.2%},\n*Val Macro Avg F1-Score: {3:>7.4},  Time: {4}'
        print(msg.format(train_loss, dev_loss, dev_acc, dev_macro_avg_f1, time_dif))
        if stateRecorder.bestValF1 < dev_macro_avg_f1: # 用Macro Average F1-Score作为判断依据
            # Classes of samples is inblance. So Macro Average F1-Score score is a better criterion than Accuracy
            print('Best Val Macro Avg F1-Score Update!')
            last_improve = epoch  # 计算上次提升 位于哪个epoch'
            stateRecorder.bestValF1 = dev_macro_avg_f1
            test(model, test_iter)
        model.train()  # 回到训练模式
        if epoch - last_improve >= require_improvement:  # 如果长期没有提高 就提前终止
            # 验证集loss超过2000batch没下降，结束训练
            print("==========\n")
            print("\nNo optimization for a long time, auto-stopping...")
            break
    return last_improve, stateRecorder.bestValF1, stateRecorder.timestamp

def test(model, test_iter):
    # test
    start_time = time.time()
    test_acc, test_f1, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)  # 计算测试集准确率，每个batch的平均损失分类报告、混淆矩阵
    msg = '\nMode of {0};  Loss: {1:>5.2},  Acc: {2:>6.2%},  Weighted-F1: {3:>6.2%}'
    print(msg.format('test', test_loss, test_acc, test_f1))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(model, data_iter, test=False):
    model.eval()  #推理模式
    loss_total = 0
    predict_all = np.array([], dtype=int)  # 存储验证集所有batch的预测结果
    labels_all = np.array([], dtype=int)  # 存储验证集所有batch的真实标签
    with torch.no_grad():   # 数据不需要计算梯度，也不会进行反向传播
        for (tests, labels) in data_iter:
            image_features = tests[0]
            text_features = tests[1]
            prompt_input_ids = tests[2]
            prompt_attention_mask = tests[3]
            outputs = model(image_features, text_features, prompt_input_ids, prompt_attention_mask)
            
            loss = stateRecorder.loss_func(outputs, labels.long())
            loss_total += loss

            _, predic = outputs.cpu().max(1)
            labels = labels.cpu()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    macro_avg_f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    weighted_recall = metrics.recall_score(labels_all, predict_all, average='weighted')
    weighted_precise = metrics.precision_score(labels_all, predict_all, average='weighted')
    weighted_f1 = 2 * weighted_recall * weighted_precise / (weighted_recall + weighted_precise)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=[f'Class {item}' for item in range(model.num_classes)],
                                               labels=range(model.num_classes), digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, weighted_f1, loss_total / len(data_iter), report, confusion
    return acc, macro_avg_f1, loss_total / len(data_iter)
