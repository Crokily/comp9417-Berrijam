import argparse
import os
from typing import Any

from PIL import Image

from common import load_image_labels, load_single_image, save_model

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch, pandas as pd, os
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from torchvision import transforms

########################################################################################################################

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir,label_col , transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # unique_labels = self.annotations.iloc[:, 1].unique()
        unique_labels = self.annotations[label_col].unique()
        unique_labels.sort()
        self.label2id = {label: id for id, label in enumerate(unique_labels)}
        self.id2label = {id: label for label, id in self.label2id.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path)
        label = self.annotations.iloc[index, 1]
        label = 1 if label == 'Yes' else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 自定义数据加载器
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        # 要先将图像转换成rgb，因为有时候是rgb有时候是rgba
        batch = [(x[0].convert('RGB'), x[1]) for x in batch]
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings

# 自定义分类器（这是训练的核心部分）
class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='binary',
            num_classes=2
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    python train.py -d "path/to/data/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models/Is Epic/"
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args

def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """

    csv_file = os.path.join(train_input_dir, train_labels_file_name)
    img_dir = train_input_dir
    
    # 图像预处理，要按模型的要求进行预处理，要数据增强也是在这里做，目前什么都没做
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),   # 调整图像大小为224x224
        # transforms.ToTensor(),           # 将图像转换为Tensor
        # transforms.Normalize(            # 归一化
        #     mean=[0.5, 0.5, 0.5],       # 根据提供信息使用这个均值
        #     std=[0.5, 0.5, 0.5]         # 和标准差来归一化图像
        # ), 这几步已经用ViTImageProcessor实现了
    ])

    # 在这里设置预训练模型的路径
    model_path = 'resources/pretrained'

    dataset = CustomDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        label_col=target_column_name,
        transform=None # 如果有数据增强，就传入transform
    )

    # 划分样本集和训练集，80%的数据用于训练，20%用于验证
    # 假设你的dataset是一个已经准备好的CustomDataset实例
    indices = list(range(len(dataset)))  # 生成索引
    np.random.shuffle(indices)  # 打乱索引

    # 定义训练数据和验证数据的大小比例
    split = int(np.floor(0.8 * len(dataset)))  # 这里我们使用80%的数据作为训练集
    train_indices, val_indices = indices[:split], indices[split:]

    # 根据索引创建数据集子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    ##### 加载模型，冻结并更改分类器 #####

    # 本地文件路径

    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(
        model_path, 
        num_labels=2,
        label2id=dataset.label2id,
        id2label=dataset.id2label
    )
    collator = ImageClassificationCollator(feature_extractor)
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collator, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, collate_fn=collator, shuffle=False, pin_memory=True)
    # 因为样本量较少，所以batch_size设为2，实现mini-batch梯度下降, shuffle=True表示每个epoch都打乱数据集

    classifier = Classifier(model, lr=1e-1) # 传入模型和学习率
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision='16-mixed', max_epochs=8) # 使用GPU训练，训练8个epoch
    trainer.fit(classifier, train_dataloader, val_dataloader)

    # 预测并输出结果
    val_batch = next(iter(val_dataloader))
    outputs = model(**val_batch)
    logits = outputs.logits
    logits_softmax = logits.softmax(1) # val_scores
    print('Preds: ', outputs.logits.softmax(1).argmax(1))
    print('Labels:', val_batch['labels'])

    # Create the output directory and don't error if it already exists.
    os.makedirs(train_output_dir, exist_ok=True)

    # # 保存模型
    model.save_pretrained(train_output_dir)
    feature_extractor.save_pretrained(train_output_dir)

if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)

########################################################################################################################
