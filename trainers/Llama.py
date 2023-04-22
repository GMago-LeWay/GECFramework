from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class LlamaTrainer:
    def __init__(self, args, config, model) -> None:
        self.args = args
        self.config = config
        self.model = model
    
    def do_train(self, train_dataloader, val_dataloader):
        raise NotImplementedError()

    def do_test(self, dataloader, mode="VAL"):
        """
        do test process, based on ids of every token(or shallow results).
        return Dict[str,value] metrics.
        The mode is a marker and does not decide test process. In some situations, TEST mode can save results.
        """
        raise NotImplementedError()

    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens. This function give final results.
        TEST mode means the data has label. if possible, print metrics.
        INFER mode means the data does not have label.
        return json results.
        """
        results = []
        for batch_data in tqdm(dataloader):        
            texts = batch_data['raw_texts']
            labels = batch_data['raw_labels']

            prompts = []
            batch_size = len(texts)

            ## add prompt and filter the long text
            generations = [""] * batch_size
            normal_text_idx = []
            for i in range(batch_size):
                # prompt = f"请改正句子中的语法错误：\n原句：{texts[i]}\n改正："
                # prompt = f"请回答输入句子的修正版本，修正所有的语法和拼写错误：\n输入语句：{texts[i]}\n改正语句："
                # prompt = f"请回答输入句子的修正版本，修正所有的语法和拼写错误，注意不是翻译：\n输入语句：{texts[i]}\n改正语句："
                # prompt = f"请修正所有的语法和拼写错误：\n原始语句：{texts[i]}\n改正语句："
                prompt = f"请改正句子中所有的语法和拼写错误：\n原句：{texts[i]}\n改正："
                if len(prompt) > self.config.max_gen_len or len(prompt) > self.config.max_seq_len/2:
                    generations[i] = f"[过长]改正：{texts[i]}"
                else:
                    prompts.append(prompt)
                    normal_text_idx.append(i)

            ## generate
            model_generations = self.model.generate(prompts, stop_words=["\n"])

            for i in range(len(model_generations)):
                generations[normal_text_idx[i]] = model_generations[i]
            # for result in results:
            #     print("🦙LLaMA:", result.strip())

            for i in range(batch_size):
                # postprocess and save
                output = generations[i].split('\n')[2]
                predict = output.split("改正：")[1]
                predict = predict.strip()

                if len(predict) > len(texts[i]) * 1.2:
                    predict = predict[:len(texts[i])]

                if mode=="TEST":
                    results.append({"src": texts[i], "predict": predict, "tgt": labels[i], "output": output})
                elif mode=="INFER":
                    results.append({"src": texts[i], "predict": predict, "output": output})
                else:
                    raise NotImplementedError()

        return results

    def save(self, save_dir):
        raise NotImplementedError()

    def load(self, save_dir):
        print("LLama cannot load other weights.")