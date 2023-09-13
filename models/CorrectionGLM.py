import torch
import logging
from peft import  PeftModel
from transformers import AutoConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.GLM.modeling_glm import *

logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=3):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == 1
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False
    

class GLMForGrammaticalCorrection(GLMPreTrainedModel):
    def __init__(self, args, settings):
        config: GLMConfig = AutoConfig.from_pretrained(
            settings.pretrained_model, 
            trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        super().__init__(config)

        # Model Settings
        self.args = args
        self.config = config
        self.settings = settings
        self.n_gpu = len(self.args.devices.split(',')) 
        # Load GLM Model
        self.pool_token = config.pool_token
        self.glm = GLMModel.from_pretrained(
            settings.pretrained_model, 
            trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        if settings.lora_model is not None:
            logger.info("loading peft model")
            self.glm = PeftModel.from_pretrained(self.glm, settings.lora_model, torch_dtype=settings.torch_dtype)
        # Sequence labeling head.
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size, dtype=settings.torch_dtype)
        classifier_dropout = settings.output_dropout_prob
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, settings.num_labels, dtype=settings.torch_dtype)
        # GLM Loss
        self.glm_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')
        # Labeling Loss
        self.labeling_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                target_ids=None,
                detection_labels=None,
                **kwargs):

        model_out = self.glm(input_ids, position_ids, attention_mask)
        outputs, lm_logits = model_out.last_hidden_states, model_out.logits

        output_for_detection = self.dropout(outputs)
        output_for_detection = torch.tanh(self.dense(output_for_detection))
        output_for_detection = self.dropout(output_for_detection)
        logits = self.out_proj(output_for_detection)
        detection_loss = self.labeling_loss(logits.view(-1, self.settings.num_labels), detection_labels.view(-1))
        lm_loss = self.glm_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
        loss = lm_loss + self.settings.detection_loss_weight * detection_loss
        return ModelOutput(loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                        logits=logits,
                        lm_logits=lm_logits,
                        hidden_states=outputs)

