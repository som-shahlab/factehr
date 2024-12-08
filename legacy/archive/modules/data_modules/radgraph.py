import os
from datasets import load_from_disk
from ipdb import set_trace
from modules.data_modules.data_module import AbstractDataModule


class RadGraph(AbstractDataModule):

    def __init__(
        self,
        root_path,
        tokenizer,
        truncation_strategy,
        model_config,
        total_samples=None,
        is_alpaca=False,
        is_medtuned=False,
    ):
        super().__init__(
            root_path,
            tokenizer,
            truncation_strategy,
            model_config,
            total_samples,
            is_alpaca,
            is_medtuned,
        )

        # change to train and also train in the next line 
        pred_set_path = os.path.join(root_path, "coral/pdac_train.hf")
        self.pred_set = load_from_disk(pred_set_path)['train'].select_columns(
            ["text"]
        )

        self.pred_set = self.pred_set.rename_column("text", "input")

        if self.total_samples is not None and self.total_samples < len(self.pred_set):
            # randomly select total_samples from pred_set
            self.pred_set = self.pred_set.shuffle(seed=42)
            self.pred_set = self.pred_set.select(range(self.total_samples))
            print("Total samples now: ", len(self.pred_set))


    