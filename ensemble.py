from difflib import SequenceMatcher

class EnsembleOnResults:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

    @staticmethod
    def __ensemble_on_one_sentence(item1, item2):
        assert item1['src'] == item2['src']
        r1 = SequenceMatcher(None, item1['src'], item2['tgt'])
        diffs1 = r1.get_opcodes()
        for diff in diffs1:
            tag, i1, i2, j1, j2 = diff

    @staticmethod
    def __ensemble_on_one_sentence(item1, item2):
        assert item1['src'] == item2['src']
        r1 = SequenceMatcher(None, item1['src'], item2['tgt'])
        diffs1 = r1.get_opcodes()
        for diff in diffs1:
            tag, i1, i2, j1, j2 = diff
