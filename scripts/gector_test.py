import os
## 探究编辑tag表之内的差别

TAGS_DIR = '/home/liwei/workspace/models/GECToR/ctc_vocab'

def read_tags(file_name):
    with open(os.path.join(TAGS_DIR, file_name)) as f:
        tags = f.readlines()
    tags = [item.strip() for item in tags]
    return tags


def analyze(tags_list1, tags_list2):
    set1 = set(tags_list1)
    set2 = set(tags_list2)

    inter = set1.intersection(set2)
    print(f"Intersection Length {len(inter)}; Set1 Length {len(set1)}; Set2 Length {len(set2)}")


analyze(read_tags('ctc_correct_tags.txt'), read_tags('mucgec_correct_tags.txt'))
