from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, AutoModelForSeq2SeqLM

def Seq2SeqModel(args, settings):
    if 'bart' in settings.pretrained_model:
        model = BartForConditionalGeneration.from_pretrained(
            settings.pretrained_model,
            # torch_dtype=settings.torch_dtype,
        )
    elif 'glm' in settings.pretrained_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.pretrained_model, trust_remote_code=True)
    else:
        raise NotImplementedError()

    return model
