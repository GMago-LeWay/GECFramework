import torch
import numpy as np
import logging
import wandb
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from collections import OrderedDict
from transformers import AutoConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.GLM.modeling_glm import *

logger = logging.getLogger(__name__)


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
            # for n, p in model.named_parameters():
            #     print(n)
            logger.info("construct peft model of glm for GEC...")
            lora_r = settings.lora_rank
            lora_alpha = lora_r*2
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
                target_modules=['query_key_value'],
                modules_to_save=['cls_dense', 'cls_out_proj']
            )
            logger.info(f"LoRA default settings: r {lora_r}, alpha {lora_alpha}")
            model = get_peft_model(model, peft_config)
        else:
            logger.info("loading peft model of glm for GEC from checkpoint...")
            model = PeftModel.from_pretrained(model, args.load, torch_dtype=settings.torch_dtype, is_trainable=True)
            logger.info("loaded peft model successfully.")
        model.print_trainable_parameters()
    return model


LOSS_CACHE = []
DETECTION_LOSS_CACHE = []
WEIGHTED_DETECTION_LOSS_CACHE = []
GLM_LOSS_CACHE = []

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
        self.steps = 0
        self.print = (args.task_mode == 'train')
        self.print_interval = self.settings.gradient_accumulation_steps * self.settings.logging_steps
        if self.print:
            project_map = {
                'mucgec': 'MuCGEC-CorrectionGLM',
                'fcgec': 'FCGEC-CorrectionGLM',
                'pretrain': 'Pretrain-CorrectionGLM',
                'c4': 'Pretrain-CorrectionGLM',
                'lang8': 'Lang8-CorrectionGLM',
                'clang8': 'Lang8-CorrectionGLM',
                'fce': 'English-CorrectionGLM',
                'nucle': 'English-CorrectionGLM',
                'hybrid': 'English-CorrectionGLM',
                'wilocness': 'WILocness-CorrectionGLM',
                'bea_dev': 'WILocness-CorrectionGLM',
            }
            wandb.init(
                # set the wandb project where this run will be logged
                project=project_map[args.dataset.lower() if args.dataset in project_map else 'CorrectionGLM'],
                # track hyperparameters and run metadata
                name=args.save_dir,
                config={
                    'mysettings': settings,
                    'myargs': args,
                }
            )
        self.n_gpu = len(self.args.devices.split(','))
        self.loss_detach = settings.loss_detach
        # Load GLM Model
        self.pool_token = config.pool_token
        self.glm = GLMModel.from_pretrained(
            settings.pretrained_model, 
            # trust_remote_code=True,
            torch_dtype=settings.torch_dtype,
        )
        # GLM Loss
        self.glm_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='sum')

        if self.settings.model_type in ['all', 'detection']:
        # Sequence labeling head.
            self.cls_dense = torch.nn.Linear(config.hidden_size, config.hidden_size, dtype=settings.torch_dtype)
            classifier_dropout = settings.output_dropout_prob
            self.dropout = torch.nn.Dropout(classifier_dropout)
            self.cls_out_proj = torch.nn.Linear(config.hidden_size, settings.num_labels, dtype=settings.torch_dtype)
            # Labeling Loss
            # self.labeling_loss = CrossEntropyLoss(ignore_index=settings.loss_ignore_id, reduction='mean')
            if settings.alpha:
                self.labeling_loss = MultiFocalLoss(num_class=settings.num_labels, alpha=settings.alpha, gamma=2, 
                                                    reduction=self.settings.loss_reduce, dtype=settings.torch_dtype, ignore_id=settings.loss_ignore_id)
            else:
                self.labeling_loss = CrossEntropyLoss(reduction=self.settings.loss_reduce, ignore_index=settings.loss_ignore_id)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                target_ids=None,
                detection_labels=None,
                **kwargs):
        global DETECTION_LOSS_CACHE, WEIGHTED_DETECTION_LOSS_CACHE, GLM_LOSS_CACHE, LOSS_CACHE
        if self.training:
            self.steps += 1
        # glm model prediction
        model_out = self.glm(input_ids, position_ids, attention_mask)
        outputs, lm_logits = model_out.last_hidden_states, model_out.logits
        lm_loss = self.glm_loss(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
        # in older version CrossEntropy with reduce='mean' is applied , but when no error sentences occurred, loss will be nan (models before 23/12/15)
        if self.settings.loss_reduce == 'mean':
            lm_loss = lm_loss / (0.1 + (target_ids != self.settings.loss_ignore_id).sum())

        # generation model
        if self.settings.model_type == 'generate':
            loss = lm_loss
            if self.print and self.training:
                GLM_LOSS_CACHE.append(lm_loss.item())
                if self.steps % self.print_interval == 0:
                    wandb.log({"glm_loss": sum(GLM_LOSS_CACHE)/len(GLM_LOSS_CACHE)})
                    GLM_LOSS_CACHE = []
            return ModelOutput(
                loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                logits={'glm': lm_logits},
            )
        # detection prediction
        else:
            output_for_detection = self.dropout(outputs)
            if self.loss_detach:
                output_for_detection = output_for_detection.detach()
            output_for_detection = torch.tanh(self.cls_dense(output_for_detection))
            output_for_detection = self.dropout(output_for_detection)
            logits = self.cls_out_proj(output_for_detection)
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
                    GLM_LOSS_CACHE.append(lm_loss.item())
                    WEIGHTED_DETECTION_LOSS_CACHE.append(self.settings.detection_loss_weight * detection_loss.item())
                    LOSS_CACHE.append(loss.item())
                    if self.steps % self.print_interval == 0:
                        wandb.log(
                            {
                                "detection_loss": sum(DETECTION_LOSS_CACHE)/len(DETECTION_LOSS_CACHE), 
                                "glm_loss": sum(GLM_LOSS_CACHE)/len(GLM_LOSS_CACHE), 
                                "weighted_detection_loss": sum(WEIGHTED_DETECTION_LOSS_CACHE)/len(WEIGHTED_DETECTION_LOSS_CACHE), 
                                "total_loss": sum(LOSS_CACHE)/len(LOSS_CACHE)
                            }
                        )
                        DETECTION_LOSS_CACHE, GLM_LOSS_CACHE, WEIGHTED_DETECTION_LOSS_CACHE, LOSS_CACHE = [], [], [], []
                # return loss and logits
                if self.training and (torch.any(torch.isnan(detection_loss)) or torch.any(torch.isnan(lm_loss))):
                    logger.info("Warning: Nan Value in loss.")
                return ModelOutput(
                    loss=loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                    logits={'glm': lm_logits, 'detection': logits},
                    # weighted_detection_loss = (self.settings.detection_loss_weight * detection_loss).unsqueeze(0) if self.n_gpu > 1 else loss,
                    # glm_loss = lm_loss.unsqueeze(0) if self.n_gpu > 1 else loss,
                )

    # 自定义load_state_dict方法
    def load_state_dict(self, state_dict, strict=True):
        # 创建一个新的state_dict用于映射旧权重名到新权重名
        new_state_dict = OrderedDict()

        # 遍历旧的state_dict
        for key, param in state_dict.items():
            # 根据你的模块名称变更来映射新的权重名
            if key.startswith('dense'):
                new_key = key.replace('dense', 'cls_dense')
            elif key.startswith('out_proj'):
                new_key = key.replace('out_proj', 'cls_out_proj')
            else:
                new_key = key  # 如果没有变化，保持原key

            # 将映射后的权重名和参数添加到新的state_dict中
            new_state_dict[new_key] = param

        # 使用新的state_dict加载模型
        super(GLMForGrammaticalCorrectionModel, self).load_state_dict(new_state_dict, strict=strict)
