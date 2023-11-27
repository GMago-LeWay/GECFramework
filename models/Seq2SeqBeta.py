from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

def Seq2SeqModel(args, settings):
    model = BartForConditionalGeneration.from_pretrained(
        settings.pretrained_model,
        # torch_dtype=settings.torch_dtype,
    )
    return model
