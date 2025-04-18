from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

MODEL_PATH = 'openai/clip-vit-large-patch14'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
cur_dir = os.path.dirname(__file__)

model = CLIPModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    torch_dtype=torch.float32 # default option
)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)


def get_single_image_feature(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs) # shape: torch.Size([1, 768])
    return features.detach().cpu().numpy()


def write_CLIPViT_feature(image_folder_path, image_features):
    for item in tqdm(range(4000)): # 3999 is the max id of English memes
        try:
            filename = 'image_ (' + str(item) + ').jpg'
            image_path = f'{image_folder_path}/{filename}'
            f = get_single_image_feature(image_path)
        except Exception as err:
            f = torch.zeros((1, 768)).numpy()
            print(err, f'==> in file [{filename}]. Filled zero tensors instead')
        finally:
            image_features = np.concatenate((image_features, f))
    return image_features


def get_E_images_feature(image_features):
    image_folder_path = os.path.join(cur_dir, '../data/Eimages/Eimages')
    image_features = write_CLIPViT_feature(image_folder_path, image_features)
    return image_features


if __name__ == '__main__':
    if not os.path.exists(os.path.join(cur_dir, '../feature/cache_E')):
        os.mkdir(os.path.join(cur_dir, '../feature/cache_E/'))

    image_features = np.empty((0, 768))
    image_features = get_E_images_feature(image_features) # get English images feature 
    print(image_features.shape)

    with open(os.path.join(cur_dir, '../feature/cache_E/id_imageFeat_ViT-L_14.pkl'), 'wb') as fp:
        pickle.dump(image_features, fp)
        print('id_imageFeat_ViT-L_14.pkl Written!')
