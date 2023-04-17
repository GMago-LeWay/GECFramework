
class Trainer:
    def __init__(self, args, config, model) -> None:
        pass
    
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
        raise NotImplementedError()

    def save(self, save_dir=None):
        raise NotImplementedError()

    def load(self, save_dir=None):
        raise NotImplementedError()


class Trainer2:
    def __init__(self, args, config, model) -> None:
        pass
    
    def do_train(self, train_dataset, val_dataset):
        raise NotImplementedError()

    def do_test(self, dataset, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        raise NotImplementedError()

    def do_infer(self, dataset, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        raise NotImplementedError()

    def save(self, save_dir):
        raise NotImplementedError()

    def load(self, save_dir):
        raise NotImplementedError()