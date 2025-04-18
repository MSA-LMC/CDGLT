import torch
from transformers import GPT2Model

gpt2_path = '/home/qianwenhao/LLM/gpt2'

# 定义图像特征处理和分类任务
class Meteor(torch.nn.Module):
    def __init__(self, num_classes=2, img_feature_dim=768):
        super(Meteor, self).__init__()
        gpt2_model = GPT2Model.from_pretrained(gpt2_path)
        # 冻结GPT-2所有参数
        for param in gpt2_model.parameters():
            param.requires_grad = False
        # 解冻LayerNorm层
        for name, module in gpt2_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
        # 再解冻Position Embedding
        gpt2_model.wpe.weight.requires_grad = True

        self.num_classes = num_classes

        self.gpt2_model = gpt2_model

        self.feature_projection = torch.nn.Linear(img_feature_dim, gpt2_model.config.n_embd)  # 图像特征转换为GPT-2的嵌入维度

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(img_feature_dim * 3, img_feature_dim),
            torch.nn.GELU()
        )

        self.prompt_embedding = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(10, gpt2_model.config.n_embd)), requires_grad=False)

        self.scale = torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(1)))
        self.classifier = torch.nn.Linear(gpt2_model.config.n_embd, num_classes)

    def SLERP(self, image_embedding, text_embedding, alpha=0.8):
        image_embedding_normed = image_embedding / image_embedding.norm(dim=1, keepdim=True)
        text_embedding_normed = text_embedding / text_embedding.norm(dim=1, keepdim=True)
        cos_sim = torch.nn.functional.cosine_similarity(image_embedding_normed, text_embedding_normed, dim=1).unsqueeze(1)
        omega = torch.arccos(cos_sim) + 1e-6  # 计算 omega
        # 球面线性插值 (SLERP)
        fused_embedding = torch.sin((1 - alpha) * omega) * image_embedding_normed + torch.sin(alpha * omega) * text_embedding_normed
        fused_embedding = fused_embedding / torch.sin(omega)
        return fused_embedding

    def forward(self, image_embedding, text_embedding, prompt_input_ids, prompt_attention_mask):
        fused_embedding = self.SLERP(image_embedding, text_embedding)
        fused_embedding = self.adapter(torch.cat([image_embedding, fused_embedding, text_embedding], dim=1))
        fused_embedding = self.feature_projection(fused_embedding).unsqueeze(1)
        prompt_embeddings =  self.prompt_embedding.unsqueeze(0).expand(image_embedding.shape[0], -1, -1)
        combined_embeddings = torch.cat([prompt_embeddings, fused_embedding], dim=1)
        outputs = self.gpt2_model(inputs_embeds=combined_embeddings)
        alpha = torch.sigmoid(self.scale)
        last_hidden_state = alpha * outputs.last_hidden_state[:, -1, :] + (1 - alpha) * outputs.last_hidden_state[:, -2, :]
        logits = self.classifier(last_hidden_state)  # 使用最后两个位置的token的线性插值的结果进行分类
        return logits

    def calculate_loss(self, logits, labels, temperature=0.1):
        # loss1; For Classification
        loss1 = torch.nn.functional.cross_entropy(logits, labels)
        return loss1
