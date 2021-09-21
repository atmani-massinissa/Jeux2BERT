import numpy as np
import torch
from torch.utils.data import Dataset


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class BertTrainDataset(Dataset):

    def __init__(self, triples, ent2input, rel2input, max_len, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        #self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        #self.count = self.count_frequency(triples)
        self.true_head, self.true_tail, self.true_rel = self.get_true_head_and_tail(self.triples)
        self.ent2input = ent2input
        self.rel2input = rel2input
        self.max_len = max_len

    def __convert_triple_to_bert_input(self, h, r, t):
        CLS, SEP = [101], [102]  # for BERT index
        #print(len(self.ent2input[h]),self.ent2input[h],len(self.rel2input[r]),self.rel2input[r],len(self.ent2input[t]),self.ent2input[t])
        #seq_len = len(self.ent2input[h])+len(self.rel2input[r])+len(self.ent2input[t])+4
        #print(seq_len)
        head = CLS + self.ent2input[h] + SEP
        seg_head = [0] * len(head)
        rel = self.rel2input[r] + SEP
        seg_rel = [1] * len(rel)
        tail = self.ent2input[t] + SEP
        seg_tail = [0] * len(tail)

        pos = head + rel + tail
        seg_pos = seg_head + seg_rel + seg_tail
        mask_pos = [1] * len(pos)

        padding = [0] * (self.max_len - len(pos))
        pos += padding
        seg_pos += padding
        mask_pos += padding

        return pos[:self.max_len], seg_pos[:self.max_len], mask_pos[:self.max_len]
    
    def __convert_triple_to_bert_input_agg(self, h, r, t, r2, t2):
        CLS, SEP = [101], [102]  # for BERT index
        head = CLS + self.ent2input[h] + SEP + self.rel2input[r] + SEP + self.ent2input[t]
        seg_head = [0] * len(head)
        rel = self.rel2input[r2] + SEP
        seg_rel = [1] * len(rel)
        tail = self.ent2input[t2] + SEP
        seg_tail = [0] * len(tail)

        pos = head + rel + tail
        seg_pos = seg_head + seg_rel + seg_tail
        mask_pos = [1] * len(pos)

        padding = [0] * (self.max_len - len(pos))
        pos += padding
        seg_pos += padding
        mask_pos += padding

        return pos[:self.max_len], seg_pos[:self.max_len], mask_pos[:self.max_len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail2,relation2=None,None
        if(len(self.triples[idx])<3):
            print("idx",idx,self.triples[idx])
        if(len(self.triples[idx])==5):
            head, relation, tail, relation2, tail2 = self.triples[idx]
        elif(len(self.triples[idx])==4):
            head, relation, tail,tail2 = self.triples[idx]  # true triple
        else:
            head, relation, tail = self.triples[idx]
        #print("idx",idx,self.triples[idx])
        if(len(self.triples[idx])==5):
            pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input_agg(head, relation, tail, relation2, tail2)
        else:
            pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head, relation, tail)
        inputs = np.array([pos])
        segment_ids = np.array([seg_pos])
        attn_masks = np.array([mask_pos])
        labels = np.array([1])
        head_ids = np.array([head])
        relation_ids = np.array([relation])
        tail_ids = np.array([tail])

        # weight
        #subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        #subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        if self.mode == 'negative':
            labels = np.array([0])
            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            labels = torch.LongTensor(labels)
            head_ids = torch.LongTensor(head_ids)
            relation_ids = torch.LongTensor(relation_ids)
            tail_ids = torch.LongTensor(tail_ids)
            return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids
        
        if self.mode == 'positive':
            labels = np.array([1])
            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            labels = torch.LongTensor(labels)
            head_ids = torch.LongTensor(head_ids)
            relation_ids = torch.LongTensor(relation_ids)
            tail_ids = torch.LongTensor(tail_ids)
            return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids
        
        
        if self.mode == 'rr_positive':
            pos2, seg_pos2, mask_pos2 = self.__convert_triple_to_bert_input(head, relation, tail2)
            labels = np.array([0])
            p_i = torch.LongTensor(pos).unsqueeze(0)
            p_s = torch.LongTensor(seg_pos).unsqueeze(0)
            p_a = torch.LongTensor(mask_pos).unsqueeze(0)
            n_i = torch.LongTensor(pos2).unsqueeze(0)
            n_s = torch.LongTensor(seg_pos2).unsqueeze(0)
            n_a = torch.LongTensor(mask_pos2).unsqueeze(0)
            #if self.negative_sample_size == 1:
            #labels = np.array([1])
            labels = torch.FloatTensor(labels)
            #seq_len = torch.LongTensor([seq_len_])
            #print("seq_len_ unsqueeze",head, relation, tail)
            return p_i, p_s, p_a, n_i, n_s, n_a, labels
                    
        negative_sample_list = np.array([])
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size/1 and self.mode not in('asym','sym'):
            m=False
            while(m==False):
                negative_sample = np.random.randint(self.nentity, size=1)
                #negative_sample = np.random.randint(self.nrelation, size=1)
                #print(negative_sample)
                if self.mode == 'head-batch':
                    if ((relation, tail) not in self.true_head.keys()):
                        mask = np.array([True])
                    else:
                        mask = np.in1d(
                            negative_sample,
                            self.true_head[(relation, tail)],
                            assume_unique=True,
                            invert=True
                        )
                elif self.mode == 'tail-batch':
                    if ((head, relation) not in self.true_tail.keys()):
                        mask = np.array([True])
                    else:
                        mask = np.in1d(
                            negative_sample,
                            self.true_tail[(head, relation)],
                            assume_unique=True,
                            invert=True
                        )
                elif self.mode == 'rel-batch' or self.mode == 'sym' or self.mode == 'asym':
                    negative_sample = np.random.randint(self.nrelation, size=1)
                    if ((head, tail) not in self.true_rel.keys()):
                        mask = np.array([True])
                    else:
                        mask = np.in1d(
                            negative_sample,
                            self.true_rel[(head, tail)],
                            assume_unique=True,
                            invert=True
                        )        
                
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                #print(mask)
                m=mask[0]
                if(m==True):
                    negative_sample = negative_sample[mask]
                    negative_sample_list = np.append(negative_sample_list,negative_sample, axis=0)
                    #print(negative_sample_list)
                    #print(negative_sample)
                    negative_sample_size += 1#negative_sample.size

        negative_sample = np.array([])
        #print(len(negative_sample_list))
        if(self.negative_sample_size>0):
            #negative_sample = negative_sample_list[:self.negative_sample_size]
            negative_sample = negative_sample_list[:self.negative_sample_size]

            #np.concatenate(negative_sample_list)[:self.negative_sample_size]
        #print(len(negative_sample))
        #if(len(negative_sample)<1):
        #    negative_sample= np.append(negative_sample,0)
        if (self.negative_sample_size>-1):
            for neg in negative_sample:
                if self.mode == 'tail-batch' and len(self.triples[idx])==5:
                    ids, seg, mask = self.__convert_triple_to_bert_input_agg(head, relation, tail, relation2, neg)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,neg))
                elif self.mode == 'head-batch' and len(self.triples[idx])==5:
                    ids, seg, mask = self.__convert_triple_to_bert_input_agg(head, relation, neg, relation2, tail2)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,tail))
                elif self.mode == 'rel-batch' and len(self.triples[idx])==5:
                    ids, seg, mask = self.__convert_triple_to_bert_input_agg(head, relation, tail, neg, tail2)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,tail))    
                elif self.mode == 'head-batch':
                    ids, seg, mask = self.__convert_triple_to_bert_input(neg, relation, tail)
                    head_ids=np.vstack((head_ids,neg))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,tail))
                elif self.mode == 'tail-batch':
                    ids, seg, mask = self.__convert_triple_to_bert_input(head, relation, neg)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,neg))
                elif self.mode == 'rel-batch':
                    ids, seg, mask = self.__convert_triple_to_bert_input(head, neg, tail)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,neg))
                    tail_ids=np.vstack((tail_ids,tail))
                elif self.mode == 'sym':
                    ids, seg, mask = self.__convert_triple_to_bert_input(tail, relation, head)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,relation))
                    tail_ids=np.vstack((tail_ids,neg))
                    labels = np.array([1])
                elif self.mode == 'asym':
                    ids, seg, mask = self.__convert_triple_to_bert_input(tail, relation, head)
                    head_ids=np.vstack((head_ids,head))
                    relation_ids=np.vstack((relation_ids,neg))
                    tail_ids=np.vstack((tail_ids,tail))
                    labels = np.array([0])
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                inputs=np.vstack((inputs,ids))
                segment_ids=np.vstack((segment_ids,seg))
                attn_masks=np.vstack((attn_masks,mask))
                if (self.negative_sample_size>1):
                    labels=np.append(labels,0)
                    
                #print("rr",len(negative_sample),len(labels))
            #print(head, relation, tail)
            #print(labels,inputs,mask)             
        else:
            #for _ in range(0,1):
            #    if self.mode == 'head-batch':
            #        ids, seg, mask = self.__convert_triple_to_bert_input(head, relation, tail)
            #        head_ids.append(head)
            #        relation_ids.append(relation)
            #        tail_ids.append(tail)
            #    elif self.mode == 'tail-batch':
            #        ids, seg, mask = self.__convert_triple_to_bert_input(head, relation, tail)
            #        head_ids.append(head)
            #        relation_ids.append(relation)
            #        tail_ids.append(tail)
            #    else:
            #        raise ValueError('Training batch mode %s not supported' % self.mode)
            #tail2!=None
            #print(len(self.triples[idx]), head, relation, tail2)
            ## rr task with negative samples from files
            ids, seg, mask = self.__convert_triple_to_bert_input(head, relation, tail2)
            inputs=np.vstack((inputs,ids))
            segment_ids=np.vstack((segment_ids,seg))
            attn_masks=np.vstack((attn_masks,mask))
            #labels=np.append(labels,0)

        if self.negative_sample_size == -1 or self.negative_sample_size == 1:
            ## rr task
            p_i = torch.LongTensor(pos).unsqueeze(0)
            p_s = torch.LongTensor(seg_pos).unsqueeze(0)
            p_a = torch.LongTensor(mask_pos).unsqueeze(0)
            n_i = torch.LongTensor(ids).unsqueeze(0)
            n_s = torch.LongTensor(seg).unsqueeze(0)
            n_a = torch.LongTensor(mask).unsqueeze(0)
            #if self.negative_sample_size == 1:
            #labels = np.array([1])
            labels = torch.FloatTensor(labels)
            #seq_len = torch.LongTensor([seq_len_])
            #print("seq_len_ unsqueeze",head, relation, tail)
            #print("rr",p_i.shape,labels.shape)
            return p_i, p_s, p_a, n_i, n_s, n_a, labels#,seq_len
        else:
            #lp task
            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            labels = torch.LongTensor(labels)
            head_ids = torch.LongTensor(head_ids)
            relation_ids = torch.LongTensor(relation_ids)
            tail_ids = torch.LongTensor(tail_ids)
            #seq_len = torch.LongTensor([seq_len_])
            #print("seq_len_",seq_len_,seq_len,np.array(seq_len_),inputs)
            #print("lp",inputs.shape,labels.shape)
            return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids#,seq_len

    @staticmethod
    def collate_fn_bert(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        #seq_len = torch.cat([_[4] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels#,seq_len

    @staticmethod
    def collate_fn_full(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        head_ids = torch.cat([_[4] for _ in data], dim=0)
        relation_ids = torch.cat([_[5] for _ in data], dim=0)
        tail_ids = torch.cat([_[6] for _ in data], dim=0)
        #seq_len = torch.cat([_[7] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids#,seq_len

    @staticmethod
    def collate_fn_rr(data):
        in_ids1 = torch.cat([_[0] for _ in data], dim=0)
        seg1 = torch.cat([_[1] for _ in data], dim=0)
        attn1 = torch.cat([_[2] for _ in data], dim=0)
        in_ids2 = torch.cat([_[3] for _ in data], dim=0)
        seg2 = torch.cat([_[4] for _ in data], dim=0)
        attn2 = torch.cat([_[5] for _ in data], dim=0)
        labels = torch.cat([_[6] for _ in data], dim=0)
        #seq_len = torch.cat([_[7] for _ in data], dim=0)
        return in_ids1, seg1, attn1, in_ids2, seg2, attn2, labels#,seq_len


    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}
        true_rel  = {}
        
        for e in triples:
            if(len(e)==3):
                head, relation, tail = e
                #if (head, relation) not in true_tail:
                #    true_tail[(head, relation)] = []
                #true_tail[(head, relation)].append(tail)
                if (relation, tail) not in true_head:
                    true_head[(relation, tail)] = []
                true_head[(relation, tail)].append(head)
                #if (head, tail) not in true_rel:
                #    true_rel[(head, tail)] = []
                #true_rel[(head, tail)].append(relation)
            #if(len(e)==5):
            #    head, relation, tail, _, _ = e
            #    '''if (head, relation) not in true_tail:
            #       true_tail[(head, relation)] = []
            #    true_tail[(head, relation)].append(tail)
            #    if (relation, tail) not in true_head:
            #       true_head[(relation, tail)] = []
            #    true_head[(relation, tail)].append(head)'''
            #    if (head, tail) not in true_rel:
            #        true_rel[(head, tail)] = []
            #    true_rel[(head, tail)].append(relation)
                
        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        #for head, relation in true_tail:
        #    true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        #for head, tail in true_rel:
        #    true_rel[(head, tail)] = np.array(list(set(true_rel[(head, tail)])))

        return true_head, true_tail, true_rel

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        #print(triples)
        if(len(triples[0])==3):
            for head, relation, tail in triples:
                if (head, relation) not in count:
                    count[(head, relation)] = start
                else:
                    count[(head, relation)] += 1
                if (tail, -relation - 1) not in count:
                    count[(tail, -relation - 1)] = start
                else:
                    count[(tail, -relation - 1)] += 1
            return count
        else:
            for head, relation, tail, tail2 in triples:
                if (head, relation) not in count:
                    count[(head, relation)] = start
                else:
                    count[(head, relation)] += 1
                if (tail, -relation - 1) not in count:
                    count[(tail, -relation - 1)] = start
                else:
                    count[(tail, -relation - 1)] += 1
                #if (tail2, -relation - 1) not in count:
                #    count[(tail2, -relation - 1)] = start
                #else:
                #    count[(tail2, -relation - 1)] += 1
            return count

class BertTrainDataset2(Dataset):

    def __init__(self, label_list,triples, ent2input, rel2input, max_len, nentity, nrelation):
        self.len = len(triples)
        self.triples = triples
        self.label_map = {label : i for i, label in enumerate(label_list)}
        #self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.ent2input = ent2input
        self.rel2input = rel2input
        self.max_len = max_len

    def __convert_triple_to_bert_input(self, h, t):
        CLS, SEP = [101], [102]  # for BERT index

        head = CLS + self.ent2input[h] + SEP
        seg_head = [0] * len(head)
        tail = self.ent2input[t] + SEP
        seg_tail = [1] * len(tail)

        pos = head  + tail
        seg_pos = seg_head + seg_tail
        mask_pos = [1] * len(pos)

        padding = [0] * (self.max_len - len(pos))
        pos += padding
        seg_pos += padding
        mask_pos += padding

        return pos[:self.max_len], seg_pos[:self.max_len], mask_pos[:self.max_len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail2=None
        if(len(self.triples[idx])<3):
            print("idx",idx,self.triples[idx])
        head, relation, tail = self.triples[idx]
        #print("idx",idx,self.triples[idx])
        pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head, tail)
        inputs = np.array([pos])
        segment_ids = np.array([seg_pos])
        attn_masks = np.array([mask_pos])
        #print(self.label_map)
        labels = np.array([self.label_map[relation]])
        head_ids = np.array([head])
        relation_ids = np.array([relation])
        tail_ids = np.array([tail])

        inputs = torch.LongTensor(inputs)
        segment_ids = torch.LongTensor(segment_ids)
        attn_masks = torch.LongTensor(attn_masks)
        labels = torch.LongTensor(labels)
        head_ids = torch.LongTensor(head_ids)
        #relation_ids = torch.LongTensor(relation_ids)
        tail_ids = torch.LongTensor(tail_ids)
        return inputs, segment_ids, attn_masks, labels, head_ids, head_ids, tail_ids

    @staticmethod
    def collate_fn_bert(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels

    @staticmethod
    def collate_fn_full(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        head_ids = torch.cat([_[4] for _ in data], dim=0)
        relation_ids = torch.cat([_[5] for _ in data], dim=0)
        tail_ids = torch.cat([_[6] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids

    @staticmethod
    def collate_fn_rr(data):
        in_ids1 = torch.cat([_[0] for _ in data], dim=0)
        seg1 = torch.cat([_[1] for _ in data], dim=0)
        attn1 = torch.cat([_[2] for _ in data], dim=0)
        in_ids2 = torch.cat([_[3] for _ in data], dim=0)
        seg2 = torch.cat([_[4] for _ in data], dim=0)
        attn2 = torch.cat([_[5] for _ in data], dim=0)
        labels = torch.cat([_[6] for _ in data], dim=0)
        return in_ids1, seg1, attn1, in_ids2, seg2, attn2, labels
    
    
class BertTrainDataset3(Dataset):

    def __init__(self, triples, ent2input, rel2input, max_len, nentity, nrelation, mode):
        self.len = len(triples)
        self.mode = mode
        self.triples = triples
        #self.label_map = {label : i for i, label in enumerate(label_list)}
        #self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.true_syn, self.true_ant = self.get_true_head_and_tail(self.triples)
        self.ent2input = ent2input
        self.rel2input = rel2input
        self.max_len = max_len

    def __convert_triple_to_bert_input(self, h):
        CLS, SEP = [101], [102]  # for BERT index
        
        head = CLS + self.ent2input[h] + SEP
        seg_head = [0] * len(head)
        pos = head
        seg_pos = seg_head
        mask_pos = [1] * len(pos)

        padding = [0] * (self.max_len - len(pos))
        pos += padding
        seg_pos += padding
        mask_pos += padding

        return pos[:self.max_len], seg_pos[:self.max_len], mask_pos[:self.max_len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail2=None
        if(len(self.triples[idx])<3):
            print("idx",idx,self.triples[idx])
        head, tail, tail2 = self.triples[idx]
        if(self.mode=="normal"):
            #print("idx",idx,self.triples[idx])
            pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head)
            inputs = np.array([pos])
            segment_ids = np.array([seg_pos])
            attn_masks = np.array([mask_pos])
            head_ids = np.array([head])

            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            head_ids = torch.LongTensor(head_ids)

            pos2, seg_pos2, mask_pos2 = self.__convert_triple_to_bert_input(tail)
            inputs2 = np.array([pos2])
            segment_ids2 = np.array([seg_pos2])
            attn_masks2 = np.array([mask_pos2])
            head_ids2 = np.array([tail])

            inputs2 = torch.LongTensor(inputs2)
            segment_ids2 = torch.LongTensor(segment_ids2)
            attn_masks2 = torch.LongTensor(attn_masks2)
            head_ids2 = torch.LongTensor(head_ids2)

            pos3, seg_pos3, mask_pos3 = self.__convert_triple_to_bert_input(tail2)
            inputs3 = np.array([pos3])
            segment_ids3 = np.array([seg_pos3])
            attn_masks3 = np.array([mask_pos3])
            head_ids3 = np.array([tail2])

            inputs3 = torch.LongTensor(inputs3)
            segment_ids3 = torch.LongTensor(segment_ids3)
            attn_masks3 = torch.LongTensor(attn_masks3)
            head_ids3 = torch.LongTensor(head_ids3)

            labels = np.array([1])
            labels = labels = torch.LongTensor(labels)

            return inputs, segment_ids, attn_masks, inputs2, segment_ids2, attn_masks2, inputs3, segment_ids3, attn_masks3, labels
        
        elif(self.mode=="syn"):
            #print("idx",idx,self.triples[idx])
            pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head)
            inputs = np.array([pos])
            segment_ids = np.array([seg_pos])
            attn_masks = np.array([mask_pos])
            head_ids = np.array([head])

            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            head_ids = torch.LongTensor(head_ids)
            negative_sample_size = 0

            while negative_sample_size < 1:           
                negative_sample = np.random.randint(self.nentity, size=1)
                if (head not in self.true_ant.keys()):
                      mask = np.array([True])
                else:
                      mask = np.in1d(
                                negative_sample,
                                self.true_ant[head],
                                assume_unique=True,
                                invert=True
                            )
                m=mask[0]
                if(m==True):
                    negative_sample = negative_sample[mask]
                    negative_sample_size += negative_sample.size

            pos2, seg_pos2, mask_pos2 = self.__convert_triple_to_bert_input(negative_sample[0])
            inputs2 = np.array([pos2])
            segment_ids2 = np.array([seg_pos2])
            attn_masks2 = np.array([mask_pos2])
            head_ids2 = np.array([negative_sample])

            inputs2 = torch.LongTensor(inputs2)
            segment_ids2 = torch.LongTensor(segment_ids2)
            attn_masks2 = torch.LongTensor(attn_masks2)
            head_ids2 = torch.LongTensor(head_ids2)

            pos3, seg_pos3, mask_pos3 = self.__convert_triple_to_bert_input(tail2)
            inputs3 = np.array([pos3])
            segment_ids3 = np.array([seg_pos3])
            attn_masks3 = np.array([mask_pos3])
            head_ids3 = np.array([tail2])

            inputs3 = torch.LongTensor(inputs3)
            segment_ids3 = torch.LongTensor(segment_ids3)
            attn_masks3 = torch.LongTensor(attn_masks3)
            head_ids3 = torch.LongTensor(head_ids3)

            labels = np.array([1])
            labels = labels = torch.LongTensor(labels)

            return inputs, segment_ids, attn_masks, inputs2, segment_ids2, attn_masks2, inputs3, segment_ids3, attn_masks3, labels
        
        elif(self.mode=="ant"):
            #print("idx",idx,self.triples[idx])
            pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head)
            inputs = np.array([pos])
            segment_ids = np.array([seg_pos])
            attn_masks = np.array([mask_pos])
            head_ids = np.array([head])

            inputs = torch.LongTensor(inputs)
            segment_ids = torch.LongTensor(segment_ids)
            attn_masks = torch.LongTensor(attn_masks)
            head_ids = torch.LongTensor(head_ids)

            pos2, seg_pos2, mask_pos2 = self.__convert_triple_to_bert_input(tail)
            inputs2 = np.array([pos2])
            segment_ids2 = np.array([seg_pos2])
            attn_masks2 = np.array([mask_pos2])
            head_ids2 = np.array([tail])

            inputs2 = torch.LongTensor(inputs2)
            segment_ids2 = torch.LongTensor(segment_ids2)
            attn_masks2 = torch.LongTensor(attn_masks2)
            head_ids2 = torch.LongTensor(head_ids2)
            negative_sample_size = 0

            while negative_sample_size < 1:           
                negative_sample = np.random.randint(self.nentity, size=1)
                if (head not in self.true_syn.keys()):
                      mask = np.array([True])
                else:
                      mask = np.in1d(
                                negative_sample,
                                self.true_syn[head],
                                assume_unique=True,
                                invert=True
                            )
                m=mask[0]
                if(m==True):
                    negative_sample = negative_sample[mask]
                    negative_sample_size += negative_sample.size

            pos3, seg_pos3, mask_pos3 = self.__convert_triple_to_bert_input(negative_sample[0])
            inputs3 = np.array([pos3])
            segment_ids3 = np.array([seg_pos3])
            attn_masks3 = np.array([mask_pos3])
            head_ids3 = np.array([negative_sample_size])

            inputs3 = torch.LongTensor(inputs3)
            segment_ids3 = torch.LongTensor(segment_ids3)
            attn_masks3 = torch.LongTensor(attn_masks3)
            head_ids3 = torch.LongTensor(head_ids3)

            labels = np.array([1])
            labels = labels = torch.LongTensor(labels)

            return inputs, segment_ids, attn_masks, inputs2, segment_ids2, attn_masks2, inputs3, segment_ids3, attn_masks3, labels

    @staticmethod
    def collate_fn_bert(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        inputs2 = torch.cat([_[3] for _ in data], dim=0)
        segment_ids2 = torch.cat([_[4] for _ in data], dim=0)
        attn_masks2 = torch.cat([_[5] for _ in data], dim=0)
        inputs3 = torch.cat([_[6] for _ in data], dim=0)
        segment_ids3 = torch.cat([_[7] for _ in data], dim=0)
        attn_masks3 = torch.cat([_[8] for _ in data], dim=0)
        labels = torch.cat([_[9] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, inputs2, segment_ids2, attn_masks2, inputs3, segment_ids3, attn_masks3, labels

    @staticmethod
    def collate_fn_full(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        head_ids = torch.cat([_[4] for _ in data], dim=0)
        relation_ids = torch.cat([_[5] for _ in data], dim=0)
        tail_ids = torch.cat([_[6] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids

    @staticmethod
    def collate_fn_rr(data):
        in_ids1 = torch.cat([_[0] for _ in data], dim=0)
        seg1 = torch.cat([_[1] for _ in data], dim=0)
        attn1 = torch.cat([_[2] for _ in data], dim=0)
        in_ids2 = torch.cat([_[3] for _ in data], dim=0)
        seg2 = torch.cat([_[4] for _ in data], dim=0)
        attn2 = torch.cat([_[5] for _ in data], dim=0)
        labels = torch.cat([_[6] for _ in data], dim=0)
        return in_ids1, seg1, attn1, in_ids2, seg2, attn2, labels
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_syn = {}
        true_ant = {}
        
        for e in triples:
                syn, syn2, ant = e
                if syn not in true_syn:
                   true_syn[syn] = []
                true_syn[syn].append(syn2)
                if syn2 not in true_syn:
                   true_syn[syn2] = []
                true_syn[syn2].append(syn)
                if syn not in true_ant:
                   true_ant[syn] = []
                true_ant[syn].append(ant)
                if ant not in true_ant:
                   true_ant[ant] = []
                true_ant[ant].append(syn)

        for syn in true_syn:
            true_syn[syn] = np.array(list(set(true_syn[syn])))
        for syn in true_ant:
            true_ant[syn] = np.array(list(set(true_ant[syn])))
        

        return true_syn, true_ant

