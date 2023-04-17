import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

class BERT(torch.nn.Module):
    def __init__(self, args, config) -> None:
        super(BERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.language_model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, texts, others=None):
        logits = self.language_model(**texts).logits
        # transpose to [batch_size, cls_num, tokens]
        return self.log_softmax(logits)


    def decode_predicted_sentence(self, text, text_ids, predict_ids):

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
            # assert truncate_len-1-pre_len == 1
        
        # revert the blanks
        for idx in blank_indexes:
            decoded_predict = decoded_predict[:idx] + ' ' + decoded_predict[idx:]

        # revert the cut text
        if len(decoded_predict) < len(raw_text):
            decoded_predict += raw_text[len(decoded_predict):]

        # retain text length
        if len(raw_text) != len(decoded_predict):
            print("Warning: Unknown reason for decoded_text longer than src text.")
            return raw_text

        return decoded_predict