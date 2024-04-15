import argparse
import os
from typing import Any

import pandas as pd
from PIL import Image
from torchvision import transforms

from common import load_model, load_predict_image_names, load_single_image

import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from transformers import ViTImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
import torch

########################################################################################################################
# 加载模型
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# 图像预处理
def preprocess_image(image_path, processor):
    image = Image.open(image_path)
    image = image.convert('RGB')  # 确保图像是RGB格式
    return processor(images=image, return_tensors='pt')

# 预测单个图像
def predict_image(model, processed_image):
    with torch.no_grad():
        outputs = model(**processed_image)
        # pred = outputs.logits.argmax(-1).item()
        pred = outputs.logits.softmax(1).argmax(1).item()
    return pred

# 从文本文件中读取图像文件名，并进行预测
def predict_from_txt(image_dir, txt_file, model, processor, output_csv, target_column_name):
    with open(txt_file, 'r') as file:
        image_files = file.read().splitlines()
    
    predictions = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        processed_image = preprocess_image(image_path, processor)
        pred = predict_image(model, processed_image)
        predictions.append([image_file, pred])
    
    # 将结果写入CSV文件
    df = pd.DataFrame(predictions, columns=['filename', target_column_name])
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args

#python predict.py 
#-d "path/to/data/Data - Is Epic Intro Full" 
#-l "Is Epic Files.txt" 
#-t "Is Epic" 
#-m "path/to/models/Is Epic/" 
#-o "path/to/predictions/Is Epic Intro Full.csv"

def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """

    # model = load_model(model_path)
    model = ViTForImageClassification.from_pretrained(trained_model_dir)  # 模型路径
    processor = ViTImageProcessor.from_pretrained(trained_model_dir)  # 模型预处理器路径

    predict_from_txt(predict_data_image_dir, predict_data_image_dir+'/'+predict_image_list, model, processor, predicts_output_csv,target_column_name)


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
