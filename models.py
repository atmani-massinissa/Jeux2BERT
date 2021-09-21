import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel,FlaubertTokenizer,FlaubertModel,FlaubertForSequenceClassification,RobertaForSequenceClassification,AlbertTokenizer,AlbertModel,AlbertForSequenceClassification
from transformers.modeling_utils import SequenceSummary


#class BertForSequenceClassification(AlbertForSequenceClassification):
class BertForSequenceClassification(FlaubertForSequenceClassification):
    
    def __init__(self, config):
        #super(FlaubertForSequenceClassification, self).__init__(config)
        super().__init__(config)
        self.transformer  = FlaubertModel(config)
        #self.transformer  = AlbertModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.dropout = nn.Dropout(config.word_keep)
        #self.dropout = nn.Dropout(config.classifier_dropout_prob) 
        self.lp_num_labels = config.lp_num_labels
        self.rp_num_labels = config.rp_num_labels
        self.lp_classifier = nn.Linear(config.hidden_size, self.lp_num_labels)
        self.rp_classifier = nn.Linear(config.hidden_size, self.rp_num_labels)
        self.mr_classifier = nn.Linear(config.hidden_size, 1)
        #self.pooling = nn.Linear(128*config.hidden_size,config.hidden_size)
        #self.layer_norm = nn.LayerNorm(config.hidden_size, self.lp_num_labels)
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity()
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, task=None,
                input_ids2=None, token_type_ids2=None, attention_mask2=None,
               input_ids3=None, token_type_ids3=None, attention_mask3=None):
        outputs=self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,output_hidden_states=True)
        #non_pooled_output = torch.mean(tuple([outputs[-1][i] for i in [-2, -1]]), dim=-1)
        #non_pooled_output = torch.div(torch.add(outputs[-1][-1],outputs[-1][-2]),2)

        #print(input_ids.shape,token_type_ids.shape,attention_mask.shape,outputs.shape)
        non_pooled_output = outputs.last_hidden_state#[0]
        #print(len(outputs),len(outputs[0]),len(outputs[1]))
        #print(len(outputs[-1]),outputs[-1][0].shape,outputs[-1][-1].shape)
        ## outputs[1][6] == outputs[0] == outputs.last_hidden_state
        #non_pooled_output = outputs[-1][-1]
        #mask_seq = torch.nonzero(attention_mask,as_tuple=False).squeeze(0)
        #pooled_output = non_pooled_output[:,0]
        #print(pooled_output.shape,attention_mask.unsqueeze(2).shape)
        #print(torch.mul(pooled_output,attention_mask.unsqueeze(2)))
        #pooled_output = torch.mul(non_pooled_output,attention_mask.unsqueeze(2))
        #print(pooled_output.shape)
        #non_pooled_output = self.dropout(non_pooled_output)
        #pooled_output = non_pooled_output[0]
        #pooled_output = self.pooling(non_pooled_output.view(-1,non_pooled_output.shape[1]*non_pooled_output.shape[2]))
        pooled_output = torch.div(torch.sum(torch.mul(non_pooled_output,attention_mask.unsqueeze(2)),1),torch.sum(attention_mask,1).unsqueeze(1))
        #print(attention_mask.shape,torch.sum(attention_mask,1))
        #print(pooled_output.shape,pooled_output,pooled_output.shape)
        #print(torch.masked_select(pooled_output,attention_mask))
        #print(pooled_output.gather(1, mask_seq))
        #pool = torch.index_select(pooled_output, 1, mask_seq)
        #print(mask_seq,pool)
        #print(outputs.last_hidden_state.shape,outputs[0].shape)        

        if task == "lp":
            pooled_output = self.dropout(pooled_output)
            logits = self.lp_classifier(pooled_output)
        elif task == "sym":
            pooled_output = self.dropout(pooled_output)
            logits1 = self.lp_classifier(pooled_output)
            
            outputs2 = self.transformer(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
            non_pooled_output2 = outputs2.last_hidden_state#[0]
            pooled_output2 = torch.div(torch.sum(torch.mul(non_pooled_output2,attention_mask2.unsqueeze(2)),1),torch.sum(attention_mask2,1).unsqueeze(1))
            pooled_output2 = self.dropout(pooled_output2)
            logits2 = self.lp_classifier(pooled_output2)
            return logits1,logits2
            
        elif task == "rp":
            #pooled_output = self.pooling(non_pooled_output.view(-1,non_pooled_output.shape[1]*non_pooled_output.shape[2]))
            #pooled_output = torch.div(torch.sum(torch.mul(non_pooled_output,attention_mask.unsqueeze(2)),1),torch.sum(attention_mask,1).unsqueeze(1))
            #pooled_output = non_pooled_output[:,0]
            pooled_output = self.dropout(pooled_output)
            logits = self.rp_classifier(pooled_output)
        elif task == "rr":
            pooled_output = self.dropout(pooled_output)
            logits1 = self.sigmoid(self.mr_classifier(pooled_output))
            #print("#################")
            #print(input_ids2.shape,input_ids.shape)
            #print(token_type_ids2.shape,token_type_ids.shape)
            #print(attention_mask2.shape,attention_mask.shape)
            outputs2 = self.transformer(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
            non_pooled_output2 = outputs2.last_hidden_state#[0]
            #pooled_output2 = non_pooled_output2[:,0]
            pooled_output2 = torch.div(torch.sum(torch.mul(non_pooled_output2,attention_mask2.unsqueeze(2)),1),torch.sum(attention_mask2,1).unsqueeze(1))
            pooled_output2 = self.dropout(pooled_output2)
            logits2 = self.sigmoid(self.mr_classifier(pooled_output2))
            #print(logits1.shape,logits2.shape)
            return logits1, logits2
        elif task =="rf":
            
            outputs2 = self.transformer(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2,output_hidden_states=True)
            non_pooled_output2 = outputs2.last_hidden_state#[0]
            non_pooled_output2 = outputs2[-1][-1]
            pooled_output2 = torch.div(torch.sum(torch.mul(non_pooled_output2,attention_mask2.unsqueeze(2)),1),torch.sum(attention_mask2,1).unsqueeze(1))
            outputs3 = self.transformer(input_ids3, token_type_ids=token_type_ids3, attention_mask=attention_mask3,output_hidden_states=True)
            non_pooled_output3 = outputs3.last_hidden_state#[0]
            non_pooled_output3 = outputs3[-1][-1]
            pooled_output3 = torch.div(torch.sum(torch.mul(non_pooled_output3,attention_mask3.unsqueeze(2)),1),torch.sum(attention_mask3,1).unsqueeze(1))
            logits1,logits2 = self.cos(pooled_output,pooled_output2),self.cos(pooled_output,pooled_output3)
            
            non_pooled_embedding = outputs[-1][0] #1,0
            pooled_embedding = torch.div(torch.sum(torch.mul(non_pooled_embedding,attention_mask.unsqueeze(2)),1),torch.sum(attention_mask,1).unsqueeze(1))
            non_pooled_embedding2 = outputs2[-1][0] #1,0
            pooled_embedding2 = torch.div(torch.sum(torch.mul(non_pooled_embedding2,attention_mask2.unsqueeze(2)),1),torch.sum(attention_mask2,1).unsqueeze(1))
            non_pooled_embedding3 = outputs3[-1][0] #1,0
            pooled_embedding3 = torch.div(torch.sum(torch.mul(non_pooled_embedding3,attention_mask3.unsqueeze(2)),1),torch.sum(attention_mask3,1).unsqueeze(1))
            
            e_logits1,e_logits2,e_logits3 = self.cos(pooled_embedding,pooled_output),self.cos(pooled_embedding2,pooled_output2),self.cos(pooled_embedding3,pooled_output3)
            #return pooled_output,pooled_output2,pooled_output3,e_logits1,e_logits2,e_logits3
            return logits1,logits2,e_logits1,e_logits2,e_logits3
        else:
            raise TypeError

        return logits
