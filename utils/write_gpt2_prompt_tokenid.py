import pickle
import torch
from transformers import GPT2Tokenizer
import os

current_dir = os.path.dirname(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpt2_path = 'openai-community/gpt2'
max_seq_length = 77

# 将文本Prompt转换为输入ID
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, device_map=device)
tokenizer.pad_token = tokenizer.eos_token

prompt_template = [ # The prompt for GPT-2 classifier
    'Sentiment Classification (happy | love | anger | sorrow | fear | hate | surprise):',
    'Sentiment Rating (slight | moderate | very):',
    'Intention Classification (interactive | expressive | entertaining | offensive | other):',
    'Offensiveness Rating (none | slight | moderate | very):',
    # 'Metaphor Detection:'
    'Metaphor Detection (literal | metaphor):',
    # 'Providing you with the memes, now identify the metaphorical ones:'
]

def get_E_text_feature(idx):
    inputs = tokenizer(prompt_template[idx], return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokenID_list = torch.cat([input_ids] * 4000, dim=0)
    mask_list = torch.cat([attention_mask] * 4000, dim=0)
    return tokenID_list, mask_list

if __name__ == '__main__':
    if not os.path.exists(os.path.join(current_dir, '../feature/cache_E')):
        os.mkdir(os.path.join(current_dir, '../feature/cache_E'))

    for idx in range(len(prompt_template)):
        tokenID_list, mask_list = get_E_text_feature(idx) # get English text feature 
        print(tokenID_list.shape, mask_list.shape)

        task_prefix = 'task' + str(idx) + '_'

        id_promptTokenID_path = os.path.join(current_dir, f'../feature/cache_E/{task_prefix}id_promptTokenID.pkl')
        id_promptMask_path = os.path.join(current_dir, f'../feature/cache_E/{task_prefix}id_promptMask.pkl')

        try:
            with open(id_promptTokenID_path, 'wb') as f:
                pickle.dump(tokenID_list, f)
            print(f'{task_prefix} tokenID_list written!')

            with open(id_promptMask_path, 'wb') as f:
                pickle.dump(mask_list, f)
            print(f'{task_prefix} mask_list written!')

        except Exception as err:
            print(err)
            break

