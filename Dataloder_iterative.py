# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import logging
import numpy as np
import json
import pickle
import torch
import math
import h5py
import random
from random import choice
# from torch._C import dtype, float32
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab

class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index, object_feature_h5,object_feat_id_to_index,
                caption_path = None, caption_max_num = None, split = None, video_names = None, question_type = None):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_video_names = video_names # dtype: str
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.object_feature_h5 = object_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.object_feat_id_to_index = object_feat_id_to_index
        self.caption_path = caption_path
        self.caption_max_num = caption_max_num
        self.caption_pool = []
        self.caption_pool_len = []
        self.video_idx2_cap_gt = {}
        self.split = split
        self.sample_caption = {}
        self.visualization = []
        self.max_word = 40
        self.different_dataset = question_type
        count = 0

        with open('data/msrvtt-qa/msrvtt-qa_val_questions.pt', 'rb') as f:
            obj = pickle.load(f)
            val_ids = obj['video_ids']
            val_id = torch.LongTensor(np.asarray(val_ids))


        if self.different_dataset == 'none':
            # matching file : question_ID video_ID caption_ID
            with open('data/msrvtt-qa/captions_pkl/full_caption_len.pkl','rb') as g:
                cap_len = pickle.load(g)

            with open(caption_path,'rb') as f:
                self.caption = pickle.load(f)  # 10000 {'video4118':[[],[]]}

                for vid,feat in self.caption.items():
                    if 'msrvtt' in self.caption_path:
                        video_idx = int(vid)
                    elif 'msvd' in self.caption_path:
                        video_idx = int(vid[3:])
                    # if video_idx in self.all_video_ids or video_idx in val_id:
                    #     continue
                    if video_idx not in self.all_video_ids:
                        continue
                    gt = []
                    max_word = self.max_word
                    for k, cap in enumerate(feat) :
                        self.visualization.append((video_idx, k))
                        if 'msvd' in self.caption_path:
                            if self.split == 'train':
                                self.sample_caption.setdefault(video_idx, []).append((cap, torch.clamp(torch.tensor(cap.shape[0]),max=max_word).data))
                            
                            gt.append(count)
                            count += 1
                            padding = torch.zeros(max_word,cap.shape[1])

                            self.caption_pool_len.append(torch.clamp(torch.tensor(cap.shape[0]),max=max_word).data)
                        
                            padding[:cap.shape[0],:] = torch.from_numpy(cap)[:max_word,:]
                            self.caption_pool.append(padding.unsqueeze(0))

                        else:
                            self.sample_caption.setdefault(video_idx, []).append((cap, cap_len[vid][k]))
                            
                            gt.append(count)
                            count += 1
                            padding = torch.zeros(max_word,cap.shape[1])

                            self.caption_pool_len.append(cap_len[vid][k])
                        
                            padding[:cap.shape[0],:] = torch.from_numpy(cap)[:max_word,:]
                            self.caption_pool.append(padding.unsqueeze(0))
                    self.video_idx2_cap_gt[str(video_idx)] = gt
                    
                self.caption_pool = torch.cat(self.caption_pool,dim=0)  # num_cap 61 768/300
                self.caption_pool_len = torch.tensor(self.caption_pool_len)
        else: # T-gif
            with open(caption_path,'r') as f:
                self.caption = json.load(f) # {'video_name': [description，[sentence_index]]}
                for vid, cap_index in self.caption.items():
                    if int(vid) not in self.all_video_ids:
                        continue
                    padding = F.pad(torch.tensor(cap_index), pad=(0,self.max_word-len(cap_index)))
                    
                    if self.split == 'train':
                        self.sample_caption.setdefault(vid, []).append((padding, torch.clamp(torch.tensor(len(cap_index)),max=self.max_word).data))
                    
                    self.caption_pool_len.append(torch.clamp(torch.tensor(len(cap_index)),max=self.max_word).data)
                    self.caption_pool.append(padding.unsqueeze(0))

                    self.video_idx2_cap_gt.setdefault(vid,[]).append(count)
                    count += 1

                self.caption_pool = torch.cat(self.caption_pool, dim=0)
                self.caption_pool_len = torch.tensor(self.caption_pool_len)
        logging.info("length of caption pool:{}".format(self.caption_pool.size()))
        logging.info("length of caption pool:{}".format(len(self.caption_pool)))

        if not np.any(ans_candidates):  # [0,0,0,0,0] -> False
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        video_name = self.all_video_names[index]
        question_idx = self.all_q_ids[index]
        ##### random sample captions
        
        if self.split == 'train':
            if self.different_dataset == 'none':
                sample_list = self.sample_caption[video_idx] #[(cap1,caplen1),(cap2,caplen2)]
                
                sample_cap, sample_cap_len = random.sample(sample_list, 1)[0]

                caption = torch.zeros(self.max_word,sample_cap.shape[1])

                caption[:sample_cap.shape[0],:] = torch.from_numpy(sample_cap)[:self.max_word,:]
                
                caption_len = torch.as_tensor(sample_cap_len)
            else:
                sample_cap, sample_cap_len = self.sample_caption[str(video_idx)][0] # ([index1,index2,...],length)
                
                caption = sample_cap
                
                caption_len = torch.as_tensor(sample_cap_len)

        ##### random sample captions
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]
        object_index = self.object_feat_id_to_index[str(video_idx)]
        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
            # if 'msrvtt' in self.app_feature_h5:
            #     Subtraction_frame = np.linspace(0, 16, num=8, endpoint=False, dtype=int)
            #     appearance_feat = appearance_feat[:, Subtraction_frame, :]
        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)
        with h5py.File(self.object_feature_h5,'r') as f_object:
            object_feat = f_object['feat'][object_index]  # (128,10,2048)
        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)
        object_feat = torch.from_numpy(object_feat).to(torch.float32)
   
        if self.split == 'train':
            return (
                    video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,
                question_len, caption, caption_len)
        else:
            return (
                    video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,
                question_len)

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)
        ##################### load caption features #####################
        caption_path = None
        dataset_name = kwargs.pop('name')
        split = kwargs.pop('split')
        caption_max_num = kwargs.pop('caption_max_num')
        #if split == 'train':
        if dataset_name == 'msrvtt-qa':
            caption_path = 'data/msrvtt-qa/captions_pkl/full_caption_features.pkl'
            #caption_path = 'data/msrvtt-qa/data/MSRVTT/structured-symlinks/aggregated_text_feats/w2v_MSRVTT.pickle'
        if dataset_name == 'msvd-qa':
            caption_path = 'data/msvd-qa/data/MSVD/structured-symlinks/aggregated_text_feats/openai-caption-full.pkl'
        if dataset_name == 'tgif-qa':
            caption_path = 'data/tgif-qa/tgif-caption/tgif_video_cap_ids.json'

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            video_names = obj['video_names']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        print('loading object feature from %s' % (kwargs['object_feat']))
        with h5py.File(kwargs['object_feat'], 'r') as object_features_file:
            object_video_ids = object_features_file['video_ids'][()]
        object_feat_id_to_index = {str(id): i for i, id in enumerate(object_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.object_feature_h5 = kwargs.pop('object_feat')
        
        self.dataset = VideoQADataset(answers, ans_candidates, ans_candidates_len, 
                                      questions, questions_len,video_ids, q_ids, 
                                      self.app_feature_h5, app_feat_id_to_index, 
                                      self.motion_feature_h5, motion_feat_id_to_index, 
                                      self.object_feature_h5,object_feat_id_to_index,
                                      caption_path, caption_max_num, split = split, 
                                      video_names = video_names,
                                      question_type = question_type,
                                      )

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
