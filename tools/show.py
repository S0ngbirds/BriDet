import torch
import argparse

def read_pth_file(file_path):
    """读取 .pth 文件并直接打印键和值（不显示张量形状等）"""
    try:
        data = torch.load(file_path, map_location='cpu')  # 加载到 CPU
        print(f"文件内容: {file_path}\n")

        if isinstance(data, dict):
            # 如果是字典（如模型 state_dict）
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            # 如果不是字典（如张量、列表等）
            print(f"数据: {data}")

    except Exception as e:
        print(f"加载失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='读取 .pth 文件的键和值')
    parser.add_argument('--file', type=str, required=True, help='.pth 文件路径')
    args = parser.parse_args()
    read_pth_file(args.file)