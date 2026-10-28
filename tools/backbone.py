import torch
from mmdet.apis import init_detector

# 加载训练好的完整模型
config_file = '../local_configs/grounding_dino_swin-t_finetune_8xb4_20e_cat.py'
checkpoint_file = '../drone_work_dir/epoch_8.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')

# 提取backbone权重
backbone_state_dict = model.backbone.state_dict()

# 保存backbone权重
torch.save(backbone_state_dict, 'epoch_8_backbone.pth')
print("Backbone weights saved successfully!")