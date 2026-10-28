# # llava_llama_descriptor.py
# import torch
# import torch.nn as nn
# import transformers
# from typing import List, Optional, Dict
# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from llava.constants import IGNORE_INDEX
# from PIL import Image

# @torch.no_grad()
# class LLaVALlamaImageDescriptor(nn.Module):
    
#     def __init__(self, 
#                  model_path: str,
#                  max_token_length: int = 512,
#                  image_size: int = 224,
#                  freeze_lm: bool = True,
#                  device: str = "cuda"):
        
#         super().__init__()
        
#         self.model_path = model_path
#         self.max_token_length = max_token_length
#         self.image_size = image_size
#         self.device = device
        
#         # 初始化LLaVA-Llama模型
#         self._init_llava_llama_model(freeze_lm)

#     def _init_llava_llama_model(self, freeze_lm: bool):
#         self.model = LlavaLlamaForCausalLM.from_pretrained(self.model_path)
#         if freeze_lm:
#             self.model.requires_grad_(False)
            
#         self.model.config.use_cache = False
        
#         # 初始化tokenizer
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(
#             self.model_path,
#             cache_dir=None,
#             model_max_length=self.max_token_length,
#             padding_side="right"
#         )
        
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
#         # 移动到设备
#         self.model.to(self.device)

#     def preprocess_image(self, images) -> torch.Tensor:
#         from torchvision import transforms
        
    
#         transform = transforms.Compose([
#             transforms.Resize((self.image_size, self.image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
#                                std=[0.26862954, 0.26130258, 0.27577711])
#         ])
        
#         processed_images = []
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         processed_images.append(transform(image))
        
#         return torch.stack(processed_images).to(self.device)

#     def create_description_prompt(self, description_type: str = "fusion") -> str:
#         prompts = {
#             "detailed": "Please provide a detailed description of this image.",
#             "concise": "Briefly describe this image.",
#             "objects": "List the main objects visible in this image.",
#             "fusion": "This is the infrared-visible light fusion task.",
#             "default": "Describe this image in detail."
#         }
        
#         return prompts.get(description_type, prompts["default"])

#     def describe_images(self, 
#                        images, 
#                        llava_cfg,
#                        description_type: str = "fusion",
#                        image_sizes: Optional[List[List[int]]] = None,
#                        tags: Optional[List[str]] = None) -> List[str]:

#         pixel_values = self.preprocess_image(images)

#         prompt = self.create_description_prompt(description_type)

#         inputs = self.tokenizer(
#             [prompt] * len(images), 
#             return_tensors="pt", 
#             padding=True,
#             truncation=True,
#             max_length=self.max_token_length
#         )

#         generation_kwargs = {
#             'input_ids': inputs['input_ids'].to(self.device),
#             'attention_mask': inputs['attention_mask'].to(self.device),
#             'images': pixel_values,
#             'max_new_tokens': llava_cfg.max_new_tokens,
#             'temperature': llava_cfg.temperature,
#             'do_sample': llava_cfg.do_sample,
#             'pad_token_id': self.tokenizer.pad_token_id,
#             'eos_token_id': self.tokenizer.eos_token_id
#         }
        

#         if image_sizes is not None:
#             generation_kwargs['image_sizes'] = image_sizes
#         if tags is not None:
#             generation_kwargs['tags'] = tags

#         with torch.no_grad():
#             outputs = self.model.generate(**generation_kwargs)
        
#         # 解码生成结果
#         descriptions = []
#         for i, output in enumerate(outputs):
#             generated_tokens = output[len(inputs['input_ids'][i]):]
#             description = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
#             descriptions.append(description.strip())
        
#         return descriptions

#     def forward(self, 
#                 images: List[Image.Image], 
#                 description_type: str = "detailed",
#                 **kwargs) -> List[str]:

#         return self.describe_images(images, description_type, **kwargs)

#     def forward_with_loss(self, 
#                          images: List[Image.Image],
#                          conversations: List[Dict],
#                          image_sizes: Optional[List[List[int]]] = None,
#                          tags: Optional[List[str]] = None) -> Dict:
#         # 预处理图像
#         pixel_values = self.preprocess_image(images)
        
#         # 准备输入数据
#         input_ids = [
#             torch.tensor(conv['input_ids'], dtype=torch.long, device=self.device) 
#             for conv in conversations
#         ]
#         labels = [
#             torch.tensor(conv['labels'], dtype=torch.long, device=self.device) 
#             for conv in conversations
#         ]
        
#         # 填充序列
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=IGNORE_INDEX)
#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
#         # 前向传播参数
#         forward_kwargs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'images': pixel_values,
#             'labels': labels,
#             'return_dict': True
#         }
        
#         # 添加可选参数
#         if image_sizes is not None:
#             forward_kwargs['image_sizes'] = image_sizes
#         if tags is not None:
#             forward_kwargs['tags'] = tags
        
#         # 前向传播
#         outputs = self.model(**forward_kwargs)
        
#         return {
#             'loss': outputs.loss,
#             'logits': outputs.logits
#         }



# def create_llama_image_descriptor(model_path: str = "liuhaotian/llava-llama-7b", **kwargs) -> LLaVALlamaImageDescriptor:
#     return LLaVALlamaImageDescriptor(model_path, **kwargs)


# def load_and_describe_image(image_path: str, 
#                            descriptor: LLaVALlamaImageDescriptor,
#                            description_type: str = "detailed",
#                            **kwargs) -> str:
#     image = Image.open(image_path).convert('RGB')
#     return descriptor.describe_single_image(image, description_type, **kwargs)

