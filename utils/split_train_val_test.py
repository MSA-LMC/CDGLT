import pandas as pd
from tqdm import tqdm
import os

cur_dir = os.path.dirname(__file__)

task_names = ["sentiment category", "sentiment degree", "intention detection", "offensiveness detection", "metaphor occurrence"]

df = pd.read_csv(os.path.join(cur_dir, f'../data/label_E.csv'), encoding='utf-8')

# df_train = pd.read_csv(os.path.join(cur_dir, '../data/train.csv'), encoding='utf-8')
# df_val = pd.read_csv(os.path.join(cur_dir, '../data/val.csv'), encoding='utf-8')
# df_test = pd.read_csv(os.path.join(cur_dir, '../data/test.csv'), encoding='utf-8')

# df_train = df.iloc[:int(len(df) * 0.6), :]
# df_val = df.iloc[int(len(df) * 0.6):int(len(df) * 0.8), :]
# df_test = df.iloc[int(len(df) * 0.8):, :]

df_train = pd.read_csv(os.path.join(cur_dir, '../data/avg_train_label_E.csv'), encoding='utf-8')
df_val = pd.read_csv(os.path.join(cur_dir, '../data/avg_val_label_E.csv'), encoding='utf-8')
df_test = pd.read_csv(os.path.join(cur_dir, '../data/avg_test_label_E.csv'), encoding='utf-8')

def hanlder_split(task_id):
    train_val_test_dfs = [df_train, df_val, df_test]
    train_val_test_str = ['SintTrain6', 'SintVal2', 'SintTest2']
    train_val_test_str = [f'task{task_id}_{item}' for item in train_val_test_str]

    saved_dir = os.path.join(cur_dir, '../data/E_split')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    for k in range(len(train_val_test_dfs)):
        # file_names = train_val_test_dfs[k].loc[:, 'file_name']
        file_names = train_val_test_dfs[k].loc[:, 'images_name']
        tmp_df = pd.DataFrame()
        for f_name in tqdm(file_names):
            f_name = f_name[2:]
            idx = int(f_name.split('image_ (')[1][:-5]) # get image id from image file name
            try:
                tmp_label = df[df['file_name'] == f_name][task_names[task_id]].values[0]
            except Exception:
                print(f'{f_name} is missing')
                continue
            real_label = 0
            if task_id < 3:
                real_label = int(tmp_label[0]) - 1
            elif task_id == 3:
                real_label = int(tmp_label[0])
            elif task_id == 4:
                real_label = tmp_label
            line = pd.DataFrame({'id': [idx], 'label': [real_label]})
            tmp_df = pd.concat([tmp_df, line], ignore_index=True)
        print(f'The {train_val_test_str[k]} shape: {tmp_df.shape}')
        tmp_df.to_csv(os.path.join(saved_dir, f'{train_val_test_str[k]}.csv'), index=False, header=False)
    print(f'Task {task_names[task_id]} finish')
    print('\n')
    


def split_train_val_test(task_id):
    hanlder_split(task_id)

if __name__ == '__main__':
    # 0: sentiment category, 1: sentiment degree, 2: intention detection, 3: offensiveness detection, 4: metaphor occurrence
    for i in range(len(task_names)):
        split_train_val_test(task_id=i)

