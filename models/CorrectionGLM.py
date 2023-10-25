import torch
import numpy as np
import logging
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
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


class MultiFocalLoss(torch.nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean', dtype=torch.float32, ignore_id=-100):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.ignore_id = ignore_id
        self.reduction = reduction
        self.smooth = 1e-4
        self.gamma = gamma
        self.alpha = alpha
        self.average_eps = 1.
        if alpha is None:
            self.alpha = torch.ones(num_class, dtype=dtype) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class, dtype=dtype)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha, dtype=dtype)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        '''
        logit: [SEQ_LENGTH, N_CLASSES]
        target: [SEQ_LENGTH]
        '''
        # probability transform
        prob = F.softmax(logit, dim=-1)
        ori_shp = target.shape
        target = target.view(-1, 1)

        ignore_mask = (target == self.ignore_id) * 1
        temp_target = target - ignore_mask*self.ignore_id    # convert ignore_id position with 0

        prob = prob.gather(1, temp_target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha = self.alpha.to(device=temp_target.device)
        alpha_weight = alpha[temp_target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
        loss_mask =  (1.-ignore_mask).squeeze(-1)
        loss = loss * loss_mask

        if self.reduction == 'mean':
            loss = loss.sum() / (self.average_eps + loss_mask.sum())
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


def GLMForGrammaticalCorrection(args, settings):
    model = GLMForGrammaticalCorrectionModel(args, settings)
    if settings.use_lora:
        if args.load is None:
            logger.info("construct peft model of glm for GEC...")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False, 
                r=8, lora_alpha=32, lora_dropout=0.1,
                target_modules=['query_key_value']
            )
            model = get_peft_model(model, peft_config)
        else:
            logger.info("loading peft model of glm for GEC from checkpoint...")
            model = PeftModel.from_pretrained(model, settings.lora_model, torch_dtype=settings.torch_dtype)
        model.print_trainable_parameters()
    return model


class GLMForGrammaticalCorrectionModel(GLMPreTrainedModel):
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
        self.loss_detach = settings.loss_detach
        # Load GLM Model
        self.pool_token = config.pool_token
        self.glm = GLMModel.from_pretrained(
            settings.pretrained_model, 
            # trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        # Sequence labeling head.
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size, dtype=settings.torch_dtype)
        classifier_dropout = settings.output_dropout_prob
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, settings.num_labels, dtype=settings.torch_dtype)
        # GLM Loss
        self.glm_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')
        # Labeling Loss
        # self.labeling_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')
        self.labeling_loss = MultiFocalLoss(num_class=settings.num_labels, alpha=settings.alpha, gamma=2, 
                                            reduction='mean', dtype=settings.torch_dtype, ignore_id=settings.loss_ignore_id)
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
        if self.loss_detach:
            output_for_detection = output_for_detection.detach()
        output_for_detection = torch.tanh(self.dense(output_for_detection))
        output_for_detection = self.dropout(output_for_detection)
        logits = self.out_proj(output_for_detection)
        detection_loss = self.labeling_loss(logits.view(-1, self.settings.num_labels), detection_labels.view(-1))
        lm_loss = self.glm_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
        loss = lm_loss + self.settings.detection_loss_weight * detection_loss
        return ModelOutput(
            loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
            logits=(lm_logits, logits),
        )
