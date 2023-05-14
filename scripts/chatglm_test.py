############################## infer test ##########################3

def infer_test():
    # 加载模型
    model_path = "/home/liwei/workspace/models/chatglm-6b"
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()

    prompt = ['\n如何制作宫保鸡丁\n', '\n蔡徐坤是谁\n']

    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    outputs = model.generate(
        input_ids = inputs["input_ids"].cuda(), 
        attention_mask = inputs['attention_mask'].cuda(),
        max_length = 512,
    )

    for output in outputs:
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        print(output_text)
