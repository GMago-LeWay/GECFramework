from data import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
    save_dir = os.path.join(DATA_ROOT_DIR, "FangZhengDapei")
    joint_save_dir = os.path.join(save_dir, "stg_joint")
    if not os.path.exists(joint_save_dir):
        os.makedirs(joint_save_dir)
    limit = {"train": 550000, "valid": 11000, "test": 11000}
    data_items = []
    with open(os.path.join(DATA_ROOT_DIR, "FangZhengAugment", "nonhgm_train_dapei.txt"), 'r') as f:
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

def prepocess_mucgec():
    dataset_name = 'mucgec'
    config = Config('seq2seq', dataset_name, False).get_config()
    data = get_data(dataset_name, 'seq2seq')(None, config)
    data.process_raw_file()

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

def split_test_data_to_new_dataset(dataset_dir='../datasets/FangZhengSpell', new_dataset_dir='../datasets/FangZhengSpellv2', train_proportion=0.5, seed=20):
    setup_seed(seed=seed)
    original_data = json.load(open(os.path.join(dataset_dir, 'test.json')))
    print(f"Using test data from {dataset_dir} (length {len(original_data)})...")
    train_data, test_data = random_split(original_data, [train_proportion, 1-train_proportion])
    train_data, test_data = list(train_data), list(test_data)
    if not os.path.exists(new_dataset_dir):
        os.makedirs(new_dataset_dir)
    json.dump(train_data, open(os.path.join(new_dataset_dir, 'train.json'), 'w'), ensure_ascii=False, indent=4)
    json.dump(train_data, open(os.path.join(new_dataset_dir, 'valid.json'), 'w'), ensure_ascii=False, indent=4)
    json.dump(test_data, open(os.path.join(new_dataset_dir, 'test.json'), 'w'), ensure_ascii=False, indent=4)
    description = {"source_dataset": os.path.basename(dataset_dir), "sample_description": f"Sample {train_proportion} source dataset for train.json and valid.json, {1-train_proportion} for test.json."}
    json.dump(description, open(os.path.join(new_dataset_dir, 'description.json'), 'w'), ensure_ascii=False, indent=4)
    print(description)
    print(f"Save to {new_dataset_dir}")

from utils.ChERRANT.parallel_to_m2 import to_m2

def cherrant_gold_labels(dataset_dir='', seed=20):
    setup_seed(seed=seed)
    save_dir = os.path.join(dataset_dir, 'ChERRANT')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_json = os.path.join(dataset_dir, 'train.json')
    valid_json = os.path.join(dataset_dir, 'valid.json')
    train_data = json.load(open(train_json))
    valid_data = json.load(open(valid_json))

    def convert_data_to_m2(data):
        src_tgt_texts = [[item['text'], item['label']] for item in data]
        res = to_m2(
            ids=[item['id'] for item in data] if 'id' in data[0] else list(range(len(src_tgt_texts))),
            src_tgt_texts=src_tgt_texts,
        )
        return res
    
    train_labels = convert_data_to_m2(train_data)
    for i in range(len(train_data)):
        train_labels[i]['text'] = train_data[i]['text']
        train_labels[i]['label'] = train_data[i]['label']
    json.dump(train_labels, open(os.path.join(save_dir, 'train.json'), 'w'), ensure_ascii=False, indent=4)

    valid_labels = convert_data_to_m2(valid_data)
    for i in range(len(valid_data)):
        valid_labels[i]['text'] = valid_data[i]['text']
        valid_labels[i]['label'] = valid_data[i]['label']
    json.dump(valid_labels, open(os.path.join(save_dir, 'valid.json'), 'w'), ensure_ascii=False, indent=4)


def merge_dataset(
        dataset_names_and_split=[
            ('fce', 'all'),
            ('nucle', 'train'),
            ('wilocness', 'train'),
        ], 
        load_model='correctionglm',
        valid_json_to_copy='',
        test_json_to_copy='',
        shuffle_seed=None,
        save_dir='../datasets/EnglishHybrid'
    ):
    '''
    Merge train set of dataset.
    ONLY text, label, id will be retained.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_file = os.path.join(save_dir, 'train.json')
    valid_file = os.path.join(save_dir, 'valid.json')
    test_file = os.path.join(save_dir, 'test.json')
    description_file = os.path.join(save_dir, 'description.txt')
    train_set = []

    desc = open(description_file, 'w')
    desc.write(
        json.dumps(
            {
                'train data source': dataset_names_and_split,
                'dataloader(TransformersDataset)': load_model,
                'valid data source': valid_json_to_copy,
                'test data source': test_json_to_copy,
                'shuffle_seed': shuffle_seed,
                'save directory': save_dir
            },
            ensure_ascii=False,
            indent=4,
        ) + '\n'
    )

    # get train set
    for dataset_name, split in dataset_names_and_split:
        assert split in ['train', 'all']
        dataset_name = dataset_name.lower()
        class A:
            dataset = dataset_name
            model = load_model
        args = A()
        config = Config(args.model, args.dataset, False).get_config()
        data = get_data(args.dataset, args.model)(args, config)
        dataset_map = data.get_dataset_map()

        # single split
        if split in ['train', 'valid', 'test']:
            current_dataset = dataset_map[split]
            max_idx = 0
            for i, item in enumerate(current_dataset):
                train_set.append(
                    {
                        "id": f"{i}_{dataset_name}_{split}",
                        "text": item["text"],
                        "label": item["label"],
                    }
                )
                max_idx = i
            desc.write(f"{dataset_name} {split} num: {max_idx+1}\n")

        # all mode
        if split == 'all':
            for cur_split in ['train', 'valid', 'test']:
                current_dataset = dataset_map[cur_split]
                max_idx = 0
                for i, item in enumerate(current_dataset):
                    train_set.append(
                        {
                            "id": f"{i}_{dataset_name}_{cur_split}",
                            "text": item["text"],
                            "label": item["label"],
                        }
                    )
                    max_idx = i
                desc.write(f"{dataset_name} {cur_split} num: {max_idx+1}\n")

    # shuffle and save train set
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(train_set)
    json.dump(train_set, open(train_file, 'w'), ensure_ascii=False, indent=4)

    # copy valid set and test set by instructed file
    os.system(f"cp {valid_json_to_copy} {valid_file}")
    os.system(f"cp {test_json_to_copy} {test_file}")

    desc.close()


def get_english_test_data(
        conll_m2_file='utils/m2scorer/official-2014.combined.m2', 
        bea19_input_file='../datasets/WILocness/wi+locness/test/ABCN.test.bea19.orig', 
        save_dir_list=[]):
    
    test_data_list = []

    # CoNLL14 data, item id {i}_conll14
    with open(conll_m2_file, 'r', encoding="utf-8") as f:
        idx_ex = 0
        src_sent, src_text = None, None
        for idx_line, _line in enumerate(f):
            line = _line.strip()
            if len(line) > 0:
                prefix, remainder = line[0], line[2:]
                if prefix == "S":
                    src_text = remainder
                    src_sent = remainder.split(" ")
                else:
                    pass
            else:  # empty line, indicating end of example
                assert src_text != None
                test_data_list.append({
                    "id": f'{idx_ex}_conll14',
                    "text": src_text,
                    "src_tokens": src_sent,
                })
                src_sent, src_text = None, None
                idx_ex += 1

    conll14_num = len(test_data_list)

    # BEA-19 test data(W&I Locness test data) item id {i}_BEA19
    BEA19_texts = open(bea19_input_file, 'r', encoding="utf-8").readlines()
    for i, line in enumerate(BEA19_texts):
        test_data_list.append(
            {
                "id": f'{i}_bea19',
                "text": line.strip(),
                "src_tokens": line.strip().split(" "),
            }
        )

    bea19_num = len(test_data_list) - conll14_num

    for dir in save_dir_list:
        test_file = os.path.join(dir, 'test.json')
        json.dump(test_data_list, open(test_file, 'w'), ensure_ascii=False, indent=4)

    print(f"CoNLL14 num: {conll14_num}, BEA19 num: {bea19_num}")


if __name__ == "__main__":
    # setup_seed(111)
    # preprocess_stgjoint('mucgec')
    # preprocess_seq2edit('augment')
    # process_gector_multi_append_data('fcgec')
    # split_data('augment')

    # split_test_data_to_new_dataset('../datasets/FangZhengSpell', '../datasets/FangZhengSpellv2', 1/2)
    # split_test_data_to_new_dataset('../datasets/FangZhengSpell', '../datasets/FangZhengSpellv3', 1/3)
    # split_test_data_to_new_dataset('../datasets/FangZhengGrammar', '../datasets/FangZhengGrammarv2', 1/2)
    # split_test_data_to_new_dataset('../datasets/FangZhengGrammar', '../datasets/FangZhengGrammarv3', 1/3)

    # convert_fcgec_seq2seq()
    # prepocess_mucgec()
    
    # cherrant_gold_labels(dataset_dir='../datasets/PreTrainSet')

    # process valid data

    # process test data of English GEC, first part is CoNLL14, second part is BEA19
    get_english_test_data(
        conll_m2_file='utils/m2scorer/official-2014.combined.m2', 
        bea19_input_file='../datasets/WILocness/wi+locness/test/ABCN.test.bea19.orig', 
        save_dir_list=[
            '../datasets/C4-200M',
            '../datasets/Lang8',
            '../datasets/clang8',
            '../datasets/NUCLE',
            '../datasets/WILocness',
        ]
    )

    merge_dataset(
        dataset_names_and_split=[
            ('fce', 'all'),
            ('nucle', 'train'),
            ('wilocness', 'train'),
        ], 
        load_model='correctionglm',
        valid_json_to_copy='../datasets/WILocness/valid.json',
        test_json_to_copy='../datasets/WILocness/test.json',
        shuffle_seed=20,
        save_dir='../datasets/EnglishHybrid'
    )
