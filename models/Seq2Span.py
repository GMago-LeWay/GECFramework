from transformers import BartForConditionalGeneration, BartTokenizer, AutoConfig, BartPretrainedModel
import torch
import torch.nn as nn
from torch.nn import init, LayerNorm, Linear, CrossEntropyLoss
from transformers.utils.generic import ModelOutput
import wandb
import logging

from models.CorrectionGLM import MultiFocalLoss

logger = logging.getLogger(__name__)


LOSS_CACHE = []
DETECTION_LOSS_CACHE = []
WEIGHTED_DETECTION_LOSS_CACHE = []
LM_LOSS_CACHE = []

class Seq2Span(BartPretrainedModel):
    def __init__(self, args, settings):
        config = AutoConfig.from_pretrained(
            settings.pretrained_model, 
            trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        super().__init__(config)

        # Model Settings
        self.args = args
        self.config = config
        self.settings = settings
        self.steps = 0
        self.print = (args.task_mode == 'train')
        self.print_interval = self.settings.gradient_accumulation_steps * self.settings.logging_steps
 
        if self.print:
            wandb.init(
                # set the wandb project where this run will be logged
                project='Seq2Span',
                # track hyperparameters and run metadata
                name=args.save_dir,
                config={
                    'mysettings': settings,
                    'myargs': args,
                }
            )

        
        self.n_gpu = len(self.args.devices.split(','))
        self.loss_detach = settings.loss_detach
        # Load BART Model
        self.bart = BartForConditionalGeneration.from_pretrained(
            settings.pretrained_model, 
            # trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        # LM Loss
        self.lm_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='sum')

        if self.settings.model_type in ['all', 'detection']:
        # Sequence labeling head.
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size, dtype=settings.torch_dtype)
            classifier_dropout = settings.output_dropout_prob
            self.dropout = torch.nn.Dropout(classifier_dropout)
            self.out_proj = torch.nn.Linear(config.hidden_size, settings.num_labels, dtype=settings.torch_dtype)
            # Labeling Loss
            # self.labeling_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')
            self.labeling_loss = MultiFocalLoss(num_class=settings.num_labels, alpha=settings.alpha, gamma=2, 
                                                reduction=self.settings.loss_reduce, dtype=settings.torch_dtype, ignore_id=settings.loss_ignore_id)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids=None,
                decoder_input_ids=None,
                target_ids=None,
                detection_labels=None,
                **kwargs):
        global DETECTION_LOSS_CACHE, WEIGHTED_DETECTION_LOSS_CACHE, LM_LOSS_CACHE, LOSS_CACHE
        if self.training:
            self.steps += 1
        # BART model prediction
        model_out = self.bart(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        outputs, lm_logits = model_out.encoder_last_hidden_state, model_out.logits
        lm_loss = self.lm_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
        # in older version CrossEntropy with reduce='mean' is applied , but when no error sentences occurred, loss will be nan (models before 23/12/15)
        if self.settings.loss_reduce == 'mean':
            lm_loss = lm_loss / (0.1 + (target_ids != self.settings.loss_ignore_id).sum())

        # generation model
        if self.settings.model_type == 'generate':
            loss = lm_loss
            if self.print and self.training:
                LM_LOSS_CACHE.append(lm_loss.item())
                if self.steps % self.print_interval == 0:
                    wandb.log({"lm_loss": sum(LM_LOSS_CACHE)/len(LM_LOSS_CACHE)})
                    LM_LOSS_CACHE = []
            return ModelOutput(
                loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                logits={'glm': lm_logits},
            )
        # detection prediction
        else:
            output_for_detection = self.dropout(outputs)
            if self.loss_detach:
                output_for_detection = output_for_detection.detach()
            output_for_detection = torch.tanh(self.dense(output_for_detection))
            output_for_detection = self.dropout(output_for_detection)
            logits = self.out_proj(output_for_detection)
            detection_loss = self.labeling_loss(logits.view(-1, self.settings.num_labels), detection_labels.view(-1))

            # detection model
            if self.settings.model_type == 'detection':
                # print loss
                if self.print and self.training:
                    DETECTION_LOSS_CACHE.append(detection_loss.item())
                    if self.steps % self.print_interval == 0:
                        wandb.log({"detection_loss": sum(DETECTION_LOSS_CACHE)/len(DETECTION_LOSS_CACHE)})
                        DETECTION_LOSS_CACHE = []
                # return loss and logits
                loss = detection_loss
                return ModelOutput(
                    loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                    logits={'detection': logits},
                )
            
            # hybrid model
            else:
                loss = lm_loss + self.settings.detection_loss_weight * detection_loss
                # print loss
                if self.print and self.training:
                    DETECTION_LOSS_CACHE.append(detection_loss.item())
                    LM_LOSS_CACHE.append(lm_loss.item())
                    WEIGHTED_DETECTION_LOSS_CACHE.append(self.settings.detection_loss_weight * detection_loss.item())
                    LOSS_CACHE.append(loss.item())
                    if self.steps % self.print_interval == 0:
                        wandb.log(
                            {
                                "detection_loss": sum(DETECTION_LOSS_CACHE)/len(DETECTION_LOSS_CACHE), 
                                "lm_loss": sum(LM_LOSS_CACHE)/len(LM_LOSS_CACHE), 
                                "weighted_detection_loss": sum(WEIGHTED_DETECTION_LOSS_CACHE)/len(WEIGHTED_DETECTION_LOSS_CACHE), 
                                "total_loss": sum(LOSS_CACHE)/len(LOSS_CACHE)
                            }
                        )
                        DETECTION_LOSS_CACHE, LM_LOSS_CACHE, WEIGHTED_DETECTION_LOSS_CACHE, LOSS_CACHE = [], [], [], []
                # return loss and logits
                if self.training and (torch.any(torch.isnan(detection_loss)) or torch.any(torch.isnan(lm_loss))):
                    logger.info("Warning: Nan Value in loss.")
                return ModelOutput(
                    loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                    logits={'glm': lm_logits, 'detection': logits},
                    # weighted_detection_loss = (self.settings.detection_loss_weight * detection_loss).unsqueeze(0) if self.n_gpu > 1 else loss,
                    # lm_loss = lm_loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                )


