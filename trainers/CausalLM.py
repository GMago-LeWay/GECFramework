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

        self.marker_map = {
            ',': '，',
            ';': '；',
            ':': '：',
            '(': '（',
            ')': '）',
            '?': '？',
            '!': '！',
        }

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
            for i in range(batch_size):
                # prompt = f"请进行语法错误的修改：\n原句：{texts[i]}\n{final_prompt}"
                # prompt = f"请直接改正原句中的语法错误，如果原句没有错误，直接输出原句：\n原句：{texts[i]}\n{final_prompt}"
                prompt = self.config.prompt.replace('[text]', texts[i]) + self.config.final_prompt
                if len(prompt) > 512:
                    generations[i] = f"[过长]{self.config.final_prompt}{texts[i]}"
                else:
                    prompts.append(prompt)
                    normal_text_idx.append(i)

            ## generate
            model_generations = self.model.generate(prompts)
            if self.config.chinese_marker_substitution:
                for i in range(len(model_generations)):
                    model_generations[i] = list(model_generations[i])
                    for j in range(len(model_generations[i])):
                        if model_generations[i][j] in self.marker_map:
                            model_generations[i][j] = self.marker_map[model_generations[i][j]]
                    model_generations[i] = ''.join(model_generations[i])

            for i in range(len(model_generations)):
                generations[normal_text_idx[i]] = model_generations[i]
            # for result in results:
            #     print("🦙LLaMA:", result.strip())

            # print(generations)
            for i in range(batch_size):
                # postprocess and save
                predict = generations[i].split(self.config.final_prompt)[1]
                predict = predict.strip()

                ## if the corrections...
                if len(predict) > len(texts[i]) * self.config.max_len_prop:
                    predict = predict[:len(texts[i])]
                if len(predict) < len(texts[i]) * self.config.min_len_prop:
                    predict = str(texts[i])

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
