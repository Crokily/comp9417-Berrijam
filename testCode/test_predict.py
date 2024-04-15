import torch
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from transformers import ViTImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor

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
def predict_from_txt(image_dir, txt_file, model, processor, output_csv):
    with open(txt_file, 'r') as file:
        image_files = file.read().splitlines()
    
    predictions = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        processed_image = preprocess_image(image_path, processor)
        pred = predict_image(model, processed_image)
        predictions.append([image_file, pred])
    
    # 将结果写入CSV文件
    df = pd.DataFrame(predictions, columns=['filename', 'prediction'])
    df.to_csv(output_csv, index=False)


# 主函数
def main():
    model_path = 'Data - Needs Respray - 2024-03-26\model\model.pth'  # 修改为你的模型路径
    image_dir = 'Data - Is Epic Intro 2024-03-25'  # 图像所在文件夹路径
    output_csv = image_dir+'\output_predictions.csv'  # 输出CSV文件名
    txt_file = 'Data - Is Epic Intro 2024-03-25\Is Epic Files.txt'  # txt文件路径，包含需要预测的文件名

    # model = load_model(model_path)
    model = ViTForImageClassification.from_pretrained('Data - Needs Respray - 2024-03-26\model')  # 模型路径
    processor = ViTImageProcessor.from_pretrained('Data - Needs Respray - 2024-03-26\model')  # 模型预处理器路径

    predict_from_txt(image_dir, txt_file, model, processor, output_csv)

if __name__ == "__main__":
    main()
