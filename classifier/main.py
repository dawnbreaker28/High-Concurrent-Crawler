import torch
from data_preprocessing import load_data, preprocess_data, create_datasets
from model_training import initialize_model, train_model, save_model

def main():
    data_file = './input/bbc-text.csv'  # 替换为你自己的文件路径
    output_dir = './saved_model'  # 替换为你想要保存模型的目录

    # 加载和预处理数据
    df = load_data(data_file)
    input_ids, attention_masks, labels, tokenizer = preprocess_data(df)
    train_dataset, val_dataset = create_datasets(input_ids, attention_masks, labels)
    print("load successfully")

    # 初始化并训练模型
    model, optimizer = initialize_model()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    trained_model = train_model(model, train_dataset, val_dataset, optimizer, device)
    print("model training finished")

    # 保存模型
    save_model(trained_model, tokenizer, output_dir)

if __name__ == "__main__":
    main()
