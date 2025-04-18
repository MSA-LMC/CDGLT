import pickle
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel
import os

MODEL_PATH = 'openai/clip-vit-large-patch14'
current_dir = os.path.dirname(__file__)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

max_seq_length = 77

model = CLIPModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    torch_dtype=torch.float32 # default option
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_E_text_feature():
    E_text_path = os.path.join(current_dir, '../data/E_text.csv')
    df = pd.read_csv(E_text_path, encoding='utf-8')
    df['text'] = df['text'].fillna('')
    features_list = []
    for idx in tqdm(range(4000)): # 3999 is the max id of English memes
        file_name = 'image_ (' + str(idx) + ').jpg'
        try:
            item = df[df['file_name'] == file_name]['text'].values[0]
        except Exception:
            print(f'The text of {file_name} is missing, filled with "".')
            item = ''
        inputs = tokenizer(item, return_tensors="pt", max_length=max_seq_length, padding='max_length', truncation=True)
        inputs.to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs) # shape: torch.Size([1, 768])
        features_list.append(features.squeeze(0).detach().cpu().numpy())
    return features_list

if __name__ == '__main__':
    if not os.path.exists(os.path.join(current_dir, '../feature/cache_E')):
        os.mkdir(os.path.join(current_dir, '../feature/cache_E'))

    
    textFeat_list = get_E_text_feature() # get English text feature 
    print(len(textFeat_list), textFeat_list[0].shape)

    id_textFeat_path = os.path.join(current_dir, '../feature/cache_E/id_textFeat_ViT-L_14.pkl')
    try:
        with open(id_textFeat_path, 'wb') as f:
            pickle.dump(textFeat_list, f)
        print('id_textFeat_ViT-L_14.pkl Written')
    except Exception as err:
        print(err)
