import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import math


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class BertForSequenceClassificationUserDefined(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.classifier_2 = nn.Linear(config.hidden_size, self.config.num_labels)
        self.mapping_head = MLP()
        #self.classifier_3 = nn.Linear(config.hidden_size//2, self.config.num_labels)
        #self.classifier = nn.Linear(2 * config.hidden_size, self.config.num_labels)
        self.softmax = nn.Softmax(dim = 1)
        self.init_weights()
        self.output_emebedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, e1_pos=None, e2_pos=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            e_pos_outputs.append(e_pos_output_i)
            
        e_pos_output = torch.stack(e_pos_outputs)

        self.output_emebedding = e_pos_output #e1&e2 cancat output

        e_pos_output = self.dropout(e_pos_output)
        #logits = self.classifier(e_pos_output)
        hidden = self.classifier(e_pos_output)
        #hidden_1 = self.classifier_2(hidden)
        logits = self.classifier_2(hidden)
        #logits = self.softmax(logits)
        e = self.mapping_head(e_pos_output)
        outputs = (logits,) +  outputs[2:]  # add hidden states and attention if they are here
        return e , outputs , (self.output_emebedding,)  # (loss), logits,e (hidden_states), (attentions), (self.output_emebedding)


# f_theta1
class RelationClassification(BertForSequenceClassificationUserDefined):
    def __init__(self, config):
        super().__init__(config)


# g_theta2
class LabelGeneration(BertForSequenceClassificationUserDefined):
    def __init__(self, config):
        super().__init__(config)

#mapping head map(·)
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden1 = nn.Sequential(
                        nn.Linear(in_features=2*768,
                                 out_features=384,
                                 bias=True),
                        nn.ReLU()
                       )
        self.hidden2 = nn.Sequential(
                        nn.Linear(in_features=384, 
                                 out_features=64,
                                 bias=True),
                        )
        self.l2norm = Normalize(2)

    def forward(self,x):
            fc1 = self.hidden1(x)
            fc2 = self.hidden2(fc1)
            #输出为两个隐藏层和输出层
            fc3 = self.l2norm(fc2)
            return fc3