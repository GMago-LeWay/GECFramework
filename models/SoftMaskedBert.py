"""
@File : model.py
@Description: softmasked-bert文本纠错模型
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/8/30
@IDE: Pycharm Professional
@REFERENCE: 论文：《Spelling Error Correction with Soft-Masked BERT》，
            模型构建参考自https://github.com/will-wiki/softmasked-bert，
            修改了部分代码，修改了项目架构，修复了部分bug，补充了一些缺失的模型结构，在loss和准确率计算中加入了mask，并增添了海量注释。
"""

from curses import raw
from itertools import accumulate
from turtle import forward
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class biGruDetector(nn.Module):
    """
    论文中的检测器
    """
    def __init__(self, input_size, hidden_size, num_layer=1):
        """
        类初始化
        Args:
            input_size: embedding维度
            hidden_size: gru的隐层维度
            num_layer: gru层数
        """
        super(biGruDetector, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layer,
                          bidirectional=True, batch_first=True)
        # GRU层
        self.linear = nn.Linear(hidden_size * 2, 1)
        # 线性层
        # 因为双向GRU，所以输入维度是hidden_size*2；因为只需要输出个概率，所以第二个维度是1
        self.activation = nn.Sigmoid()

    def forward(self, inp):
        """
        类call方法的覆盖
        Args:
            inp: 输入数据，embedding之后的！形如[batch_size,sequence_length,embedding_size]

        Returns:
            模型输出
        """
        rnn_output, _ = self.rnn(inp)
        # rnn输出output和最后的hidden state，这里只需要output；
        # 在batch_first设为True时，shape为（batch_size,sequence_length,2*hidden_size）;
        # 因为是双向的，所以最后一个维度是2*hidden_size。
        output = self.activation(self.linear(rnn_output))
        # sigmoid函数，没啥好说的，论文里就是这个结构
        return output
        # output维度是[batch_size, sequence_length, 1]



class GRUSoftMaskedBERT(nn.Module):
    def __init__(self, args, config) -> None:
        super(GRUSoftMaskedBERT, self).__init__()
        self.args = args
        self.config = config

        """
        初始化
        Args:
            config: 实例化的参数管理器
        """
        # 实例化BertEncoder类，即attention结构，默认num_hidden_layers=12，也可以去本地bert模型的config.json文件里修改
        # 论文里也是12，实际运用时有需要再改
        # 查了源码，BertModel这个类还有BertEmbeddings、BertEncoder、BertPooler属性，在此之前我想获得bert embeddings都是直接用BertModel的call方法的，学习了
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model)  # 加载tokenizer
        self.vocab_size = self.tokenizer.vocab_size  # 词汇量

        # BERT
        self.language_model = BertModel.from_pretrained(self.config.pretrained_model)
        self.linear = nn.Linear(self.config.embedding_size, self.vocab_size)  # 线性层，没啥好说的
        # 加载[mask]字符对应的编码
        self.mask_id = torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long).to(args.device)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # LogSoftmax就是对softmax取log

        # Detection
        self.detector_model = biGruDetector(self.config.embedding_size, self.config.hidden_size)  # 实例化检测器


    def forward(self, input_ids, input_mask):
        # Detection
        batch_inp_embedding = self.language_model.embeddings(input_ids) # 获取输入文本序列的embedding表示
        prob = self.detector_model(batch_inp_embedding) 

        # Mask and Correction
        masked_e = self.language_model.embeddings(self.mask_id)
        soft_bert_embedding = prob * masked_e + (1 - prob) * batch_inp_embedding  
        extended_mask = self.language_model.get_extended_attention_mask(input_mask, input_ids.shape)
        bert_out = self.language_model.encoder(hidden_states=soft_bert_embedding, attention_mask=extended_mask)
        h = bert_out[0] + batch_inp_embedding
        out = self.log_softmax(self.linear(h))
        return prob, out


    def decode_predicted_sentence(self, text, text_ids, predict_ids):
        # TODO: process [UNK]: replace [UNK] position with src text

        # notice that if token length in text_ids != correspond token length in predict_ids,
        # then copy id from text_ids
        sentence_len = len(text_ids)
        assert sentence_len == len(predict_ids)
        text_tokens = self.tokenizer.convert_ids_to_tokens(text_ids)
        predict_tokens = self.tokenizer.convert_ids_to_tokens(predict_ids)
        
        # for tokens whose length different from src text, copy corresponding tokens from src.
        for i in range(sentence_len):
            if len(text_tokens[i]) != len(predict_tokens[i]):
                predict_ids[i] = text_ids[i]
        decoded_predict = self.tokenizer.decode(predict_ids).replace(' ', '')

        # chinese blank -> standard blank
        text = text.replace("\u3000", " ")
        
        # record of blanks
        raw_text = str(text)          # store src text before clear blanks
        first_blank_index = text.find(' ')
        blank_indexes = []
        if first_blank_index != -1:
            for i in range(first_blank_index, len(text)):
                if text[i] == ' ':
                    blank_indexes.append(i)
        text = text.replace(" ", "")

        # substitute [UNK] as token from src.
        while decoded_predict.find("[UNK]") != -1:
            pre_len = decoded_predict.find("[UNK]")
            # identify the span of [UNK]
            decoded_next_tokens = ['[CLS]', '[UNK]', '[SEP]']
            truncate_len = pre_len + 1
            while len(decoded_next_tokens) != 4:
                truncate_len += 1
                if truncate_len > len(text):
                    break
                decoded_next_tokens = self.tokenizer.encode(text[pre_len:truncate_len])
            decoded_predict = decoded_predict.replace('[UNK]', text[pre_len:truncate_len-1], 1)
            # [UNK] must be generated from an unknown char prefix in vocab
            assert truncate_len-1-pre_len == 1
        
        # revert the blanks
        for idx in blank_indexes:
            decoded_predict = decoded_predict[:idx] + ' ' + decoded_predict[idx:]

        # revert the cut text
        if len(decoded_predict) < len(raw_text):
            decoded_predict += raw_text[len(decoded_predict):]

        # retain text length
        assert len(raw_text) == len(decoded_predict)

        return decoded_predict

            