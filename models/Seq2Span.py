from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
import torch
import torch.nn as nn

# 定义新的Bart模型，增加Encoder部分的全连接层
class BARTWithErrorClassification(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.bart = BartModel(config)
        # 增加全连接层，假设3分类任务，输出维度为3
        self.error_classifier = nn.Linear(config.hidden_size, 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        error_labels=None,
    ):
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        sequence_output = outputs[0]  # 获取Encoder的最后一层输出
        pooled_output = self.dropout(sequence_output[:, 0])  # 取第一个Token的向量作为句子的表示
        error_logits = self.error_classifier(pooled_output)  # 通过全连接层得到错误分类的 logits

        if labels is not None and error_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 计算生成损失（这里未展示）
            # ...
            # 计算错误分类损失
            error_loss = loss_fct(error_logits.view(-1, error_logits.size(-1)), error_labels.view(-1))
            return outputs + (error_loss,)

        return outputs + (error_logits,)

# 初始化模型和分词器
model = BARTWithErrorClassification.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# 准备输入数据（这里仅为示例，实际项目中应根据具体数据集进行处理）
input_ids = tokenizer("This is a test sentence with some error.", return_tensors="pt")
error_labels = torch.tensor([0, 1, 2], dtype=torch.long)  # 假设的错误标签

# 模型前向传播
outputs = model(input_ids=input_ids, error_labels=error_labels)
error_loss = outputs[3]
