from tqdm import tqdm
import os
import logging

from trainers.base import Trainer

logger = logging.getLogger(__name__)

class CausalLMTrain(Trainer):
    def __init__(self, args, config, model):
        super(CausalLMTrain, self).__init__(args, config, model)
        self.args = args
        self.config = config
        self.model = model

        logger.info("You have loaded CausalLM Trainer for LLM, but it can only do infer task.")

    def do_train(self, train_dataloader, val_dataloader):
        raise NotImplementedError()


    def do_test(self, dataloader, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        raise NotImplementedError()


    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
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
            final_prompt = '改正语句：'
            for i in range(batch_size):
                # prompt = f"请改正句子中的语法错误：\n原句：{texts[i]}\n改正："
                # prompt = f"请回答输入句子的修正版本，修正所有的语法和拼写错误：\n输入语句：{texts[i]}\n改正语句："
                prompt = f"请回答输入句子的修正版本，修正所有的语法和拼写错误，注意不是翻译：\n输入语句：{texts[i]}\n{final_prompt}"
                # prompt = f"请修正所有的语法和拼写错误：\n原始语句：{texts[i]}\n改正语句："
                # prompt = f"请改正句子中所有的语法和拼写错误：\n原句：{texts[i]}\n改正："
                if len(prompt) > 400:
                    generations[i] = f"[过长]{final_prompt}{texts[i]}"
                else:
                    prompts.append(prompt)
                    normal_text_idx.append(i)

            ## generate
            model_generations = self.model.generate(prompts)

            for i in range(len(model_generations)):
                generations[normal_text_idx[i]] = model_generations[i]
            # for result in results:
            #     print("🦙LLaMA:", result.strip())

            for i in range(batch_size):
                # postprocess and save
                predict = generations[i].split(final_prompt)[1]
                predict = predict.strip()

                if len(predict) > len(texts[i]) * 1.2:
                    predict = predict[:len(texts[i])]

                if mode=="TEST":
                    results.append({"src": texts[i], "predict": predict, "tgt": labels[i], "output": generations[i]})
                elif mode=="INFER":
                    results.append({"src": texts[i], "predict": predict, "output": generations[i]})
                else:
                    raise NotImplementedError()

        return results

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass
