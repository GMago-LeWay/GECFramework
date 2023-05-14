from data import *

def filter_stg_joint(data_list, limit_num):

    model_config = Config(model='stgjoint', dataset='fangzhengdapei').get_config()
    check_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_ROOT_DIR, "chinese-roberta-wwm-ext"))

    ## To check item for TaggerConvertor
    def _preprocess_gendata(ops: dict):
        '''
        Pre-tokenize modify labels and insert labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = check_tokenizer.convert_tokens_to_ids(check_tokenizer.tokenize(labstr))
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = check_tokenizer.convert_tokens_to_ids(check_tokenizer.tokenize(labstr))
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    Sentence = []
    Label = []   
    Correction = []
    for item in tqdm(data_list): 
        token = check_tokenizer.tokenize(TextWash.punc_wash(item['text'])) 
        sent_recycle_len = len(check_tokenizer.convert_tokens_to_string(token).replace(" ", ""))    
        sent_wash_len = len(TextWash.punc_wash(item['text']))
        if sent_wash_len != sent_recycle_len:
            continue
        try:
            opt_edit = min_dist_opt(item['text'], item['label'])  
            edit_label = [opt_edit]

            ## Check TaggerConvertor
            kwargs = {
                'sentence' : TextWash.punc_wash(item['text']),
                'ops' : _preprocess_gendata(opt_edit),
                'token' : token
            }
            tagger = TaggerConverter(model_config, auto=True, **kwargs)

            Sentence.append(item['text'])
            Correction.append(item['label'])
            Label.append(json.dumps([opt_edit], ensure_ascii=False))
            if len(Sentence) == limit_num:
                break
        except:
            print("Error While Coverting: %s; %s" % (item['text'], item['label']))

    return Sentence, Correction, Label
    

def filter_fangzhengdapei():
    save_dir = "/home/liwei/workspace/datasets/FangZhengDapei"
    joint_save_dir = os.path.join(save_dir, "stg_joint")
    if not os.path.exists(joint_save_dir):
        os.makedirs(joint_save_dir)
    limit = {"train": 550000, "valid": 11000, "test": 11000}
    data_items = []
    with open(os.path.join("/home/liwei/workspace/datasets/FangZhengAugment", "nonhgm_train_dapei.txt"), 'r') as f:
        for item in f.readlines():
            item_content = item.split()
            text, label = item_content[0].strip(), item_content[1].strip()
            if len(text) == len(label) and len(text) < 200:
                data_items.append({"text": text, "label": label})
    random.shuffle(data_items)
    filter_list = {
        "train": data_items[:1000000],
        "valid": data_items[1000000:1020000],
        "test": data_items[1020000:1040000],
    }

    for split in filter_list:
        sentences, corrections, labels = filter_stg_joint(data_list=filter_list[split], limit_num=limit[split])
        assert len(sentences) == len(corrections) == len(labels)
        json_res = [{"text": sentences[i], "label": corrections[i]} for i in range(len(sentences))]
        df = pd.DataFrame({"Sentence": sentences, "Label": labels})
        with open(os.path.join(save_dir, f"{split}.json"), 'w') as f:
            json.dump(json_res, f, ensure_ascii=False, indent=4)
        df.to_csv(os.path.join(joint_save_dir, f"{split}.csv"), index=False)


def preprocess(dataset_name):
    config = Config(None, dataset_name, False).get_config()
    data = get_data(dataset_name)(None, config)
    data.process_raw_file()

def preprocess_seq2edit(dataset_name):
    config = Config('seq2edit', dataset_name, False).get_config()
    data = get_data(dataset_name, 'seq2edit')(None, config)
    data.preprocess_data()

def preprocess_stgjoint(dataset_name):
    ### Use it when dataset is already split.
    config = Config(None, dataset_name, False).get_config()
    data: TextLabelDataset = get_data(dataset_name)(None, config)
    data.process_data_to_STG_Joint()

def split(dataset_name):
    ## generate split dataset
    config = Config(None, dataset_name, False).get_config()
    data: TextLabelDataset = get_data(dataset_name)(None, config)
    data.train_val_test_data()

def convert_fcgec_seq2seq():
    config = Config('stgjoint', 'fcgec', False).get_config()
    data = get_data('fcgec', 'stgjoint')(None, config)
    data.convert_seq2seq()

def process_gector_multi_append_data(dataset):
    config = Config('gector', dataset, False).get_config()
    data = get_data(dataset, 'gector')(None, config)
    data.split_multi_append()

def split_data(dataset):
    config = Config(None, dataset, False).get_config()
    data: TextLabelDataset = get_data(dataset)(None, config)
    data.train_val_test_data()

if __name__ == "__main__":
    # preprocess_stgjoint('mucgec')
    # preprocess_seq2edit('augment')
    # convert_fcgec_seq2seq()
    # process_gector_multi_append_data('pretrain')
    split_data('augment')
