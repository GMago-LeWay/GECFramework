from transformers import BartForConditionalGeneration


def Seq2SeqModel(args, config):
    return BartForConditionalGeneration.from_pretrained(config.pretrained_model)
