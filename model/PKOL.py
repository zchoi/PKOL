import numpy as np
from torch.nn import functional as F
from .utils import *

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim*2, module_dim, bias=False)

        self.cat = nn.Linear(3 * module_dim, module_dim*2)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):

        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = q_proj * v_proj

        v_distill = torch.cat([v_q_cat, visual_feat],dim=-1)

        v_distill = self.activation(self.cat(v_distill))

        return v_distill


class Global_FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(Global_FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.m_proj = nn.Linear(module_dim*4, module_dim, bias=True)
        self.o_proj = nn.Linear(module_dim*4, module_dim, bias=True)

        self.global_query = nn.Linear(module_dim*2, module_dim, bias=True)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)
        
        self.global_att = nn.Linear(module_dim, 1)
        self.obj_att = nn.Linear(module_dim, 1)
        self.final_att = nn.Linear(module_dim,1)


    def forward(self, motion_feat, obj_feat, query):
        '''
            motion_feat: batch_size num_clip 2048
            obj_feat:    batch_size num_frame num_obj 2048
            query:       batch_size module
        
        '''
        bs, num_frame, num_obj, visual_dim = obj_feat.size()
        _, num_clip, _ = motion_feat.size()
        obj_feat = self.dropout(obj_feat)
        mot_feat = self.dropout(motion_feat)

        m_proj = self.activation(self.m_proj(mot_feat))     # batch_size num_clip module
        o_proj = self.activation(self.o_proj(obj_feat))     # batch_size num_frame num_obj module
        query = self.activation(self.global_query(query))   # batch_size module
  
        m_att = self.global_att(m_proj*query.unsqueeze(1))   # batch_size num_clip 1
        m_score = F.softmax(m_att, dim=1)  # batch_size num_clip 1

        m = (m_score * m_proj).sum(1) # batch_size module
        # print((m_score * m_proj).unsqueeze(-2).repeat(1,1,num_frame//num_clip,1).reshape(bs,-1,self.module_dim).unsqueeze(-2).size())
        o_att = self.obj_att(o_proj*query.unsqueeze(1).repeat(1, num_frame, 1).unsqueeze(2))
        # o_att = self.obj_att(o_proj*(m_score * m_proj).unsqueeze(-2).repeat(1,1,num_frame//num_clip,1).reshape(bs,-1,self.module_dim).unsqueeze(-2))

        o_score = F.softmax(o_att, dim=1)  # batch_size num_frame num_obj 1

        o = (o_score * o_proj).sum(1) # batch_size num_obj module

        global_o = m.unsqueeze(1)*o
        # global_o = o * query.unsqueeze(1)

        final_att = self.final_att(global_o)
        final_score = F.softmax(final_att, dim=1)
        final = (final_score * global_o).sum(1) # batch_size module

        # print(final.size(),m.size())
        g = torch.cat([final, m],dim=-1) # batch_size module*2

        return g


class Prospect_Background_aggregation(nn.Module):
    def __init__(self, module_dim = 512):
        super(Prospect_Background_aggregation,self).__init__()
        self.module_dim = module_dim

        self.a_proj = nn.Linear(module_dim*4, module_dim, bias=True)
        self.o_proj = nn.Linear(module_dim*4, module_dim, bias=True)
        self.global_query = nn.Linear(module_dim*2, module_dim, bias=True)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)
        
        self.global_att = nn.Linear(module_dim, 1)
        self.obj_att = nn.Linear(module_dim, 1)
        self.final_att = nn.Linear(module_dim,1)

    def forward(self, obj_feat, app_feat, query):
        '''
            obj_feat: batch_size num_frame num_obj 2048
            app_feat: batch_size num_clip num_frame 2048
            query:    batch_size module
        '''
        obj_feat = self.dropout(obj_feat)
        app_feat = self.dropout(app_feat)
        
        obj_feat = self.activation(self.o_proj(obj_feat))
        app_feat = self.activation(self.a_proj(app_feat))
        query =self.activation(self.global_query(query))

        bs, num_frame, num_obj, visual_dim = obj_feat.size()

        bs, num_clip, num_clip_frame, vis_dim = app_feat.size()

        a_proj = app_feat.reshape(bs, -1, vis_dim) # bs num_frame 2048
        
        a_att = self.global_att(a_proj*query.unsqueeze(1))   # batch_size n 1
        a_score = F.softmax(a_att, dim=1)  # batch_size num_frame 1

        a = (a_score * a_proj).sum(1) # batch_size module
        
        o_att = self.obj_att(obj_feat*query.unsqueeze(1).repeat(1, num_frame, 1).unsqueeze(2)) 
        # o_att = self.obj_att(obj_feat*(a_score * a_proj).unsqueeze(-2))     
        o_score = F.softmax(o_att, dim=1)  # batch_size num_frame num_obj 1

        o = (o_score * obj_feat).sum(1) # batch_size num_obj module

        global_o = a.unsqueeze(1)*o

        # global_o = o * query.unsqueeze(1)

        final_att = self.final_att(global_o)
        final_score = F.softmax(final_att, dim=1)
        final = (final_score * global_o).sum(1) # batch_size module
        # print(final.size(),m.size())
        app = torch.cat([final, a],dim=-1) # batch_size module*2

        return app

class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))

        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class captionLinguistic(nn.Module):
    def __init__(self, caption_dim, rnn_dim=512, module_dim=512, bidirectional=True):
        super(captionLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Linear(caption_dim, rnn_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(caption_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, caption, caption_len):
        """
        Args:
            caption: [Tensor] (batch_size, num_cap, max_question_length, cap_dim)
            caption_len: [Tensor] (batch_size, num_cap)
        return:
            caption representation [Tensor] (batch_size, num_cap, module_dim)
        """
        
        caption_embedding = caption
        embed = self.tanh(self.embedding_dropout(caption_embedding))

            
        embed_candi = nn.utils.rnn.pack_padded_sequence(embed, caption_len.cpu(), batch_first=True,
                                                enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (caption_embedding, _) = self.encoder(embed_candi)
        if self.bidirectional:
            caption_embedding = torch.cat([caption_embedding[0], caption_embedding[1]], -1)
        caption_embedding = self.question_dropout(caption_embedding)
        
        return caption_embedding


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        #out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512, caption_flag = False):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.caption_flag = caption_flag

        if not self.caption_flag:
            self.classifier = nn.Sequential(nn.Dropout(0.15),
                                            nn.Linear(module_dim * 7, module_dim),
                                            nn.ELU(),
                                            nn.BatchNorm1d(module_dim),
                                            nn.Dropout(0.15),
                                            nn.Linear(module_dim, 1))
        else:
            self.classifier = nn.Sequential(nn.Dropout(0.15),
                                            nn.Linear(module_dim * 7, module_dim),
                                            nn.ELU(),
                                            nn.BatchNorm1d(module_dim),
                                            nn.Dropout(0.15),
                                            nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding, caption_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        if self.caption_flag:
            out = torch.cat([out,caption_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512, caption_flag=True):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)
        self.caption_flag = caption_flag
        if not caption_flag:
            self.regression = nn.Sequential(nn.Dropout(0.15),
                                            nn.Linear(module_dim * 2, module_dim),
                                            nn.ELU(),
                                            nn.BatchNorm1d(module_dim),
                                            nn.Dropout(0.15),
                                            nn.Linear(module_dim, 1))
        else:
            self.regression = nn.Sequential(nn.Dropout(0.15),
                                nn.Linear(module_dim * 6, module_dim),
                                nn.ELU(),
                                nn.BatchNorm1d(module_dim),
                                nn.Dropout(0.15),
                                nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding, caption_embedding = None):
        question_embedding = self.question_proj(question_embedding)
        if self.caption_flag:
            out = torch.cat([visual_embedding, question_embedding,caption_embedding], 1)
            out = self.regression(out)
        else:
            out = torch.cat([visual_embedding, question_embedding], 1)
            out = self.regression(out)

        return out


class PKOL_Net(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type, caption_dim, topk, corpus, corpus_len, patch_number, cap_vocab = None, visualization= False):
        super(PKOL_Net, self).__init__()

        self.visualization = visualization
        self.topk = topk
        self.patch_number = patch_number

        self.corpus = corpus
        self.corpus_len = corpus_len
        self.question_type = question_type
        
        self.feature_aggregation_global = FeatureAggregation(module_dim)
        self.feature_aggregation_PB = FeatureAggregation(module_dim)
        self.Global_FeatureAggregation = Global_FeatureAggregation(module_dim)
        self.Prospect_Background_aggregation = Prospect_Background_aggregation(module_dim)
        
        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim, caption_flag=True)
            self.ffc =  nn.Sequential(
                nn.Linear(module_dim*4,module_dim*2),
                nn.Dropout(p=0.2),
                nn.Tanh()
                )
        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim, caption_flag=True)
            self.classifier = nn.Sequential(nn.Dropout(0.15),
                                nn.Linear(module_dim * 2, module_dim),
                                nn.ELU(),
                                nn.BatchNorm1d(module_dim),
                                nn.Dropout(0.15),
                                nn.Linear(module_dim, 1))
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.caption_unit = captionLinguistic(caption_dim=caption_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)

        #########################################
            self.classifier = nn.Sequential(nn.Dropout(0.15),
                                    nn.Linear(module_dim * 2, module_dim),
                                    nn.ELU(),
                                    nn.BatchNorm1d(module_dim),
                                    nn.Dropout(0.15),
                                    nn.Linear(module_dim, self.num_classes))
        self.att = nn.Sequential(
            nn.Linear(module_dim,1),
            nn.Dropout(p=0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(module_dim*6,module_dim*2),
            nn.Dropout(p=0.2),
            nn.Tanh()
        )

        self.softmax = nn.Softmax(dim=-1)
   
        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, video_object_feat, question,
                question_len, similarity=None, corpus=None, corpus_len=None):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            video_object_feat: [Tensor] (batch_size, num_objs, obj_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
            simility: [Tensor] (batch_size, num_corpus)
            corpus: [Tensor] train: (num_cap, num_word cap_dim) / val: (None) | [tgif] train: (num_cap, num_word) / val: (None)
            corpus_len: [Tensor] train: (num_cap, ) / val: (None) | [tgif] train: (num_cap, ) / val: (None, )
        return:
            logits.
        """ 
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len) # batch_size module_dim
 
            # corpus retrieve

            caption_tensor = corpus
            sort_, index_ = similarity.sort(1,descending=True) # batch_size num_cap
            
            index_ = index_[:,:self.topk].contiguous().view(-1, 1)  # batch_size*topk
            
            caption_tensor_list = caption_tensor[index_].contiguous().view(batch_size, self.topk, -1) # batch_size topk module_dim 
            
            #####################caption&question-attention#####################
            caption_awear_q = caption_tensor_list*question_embedding.unsqueeze(1).repeat(1,caption_tensor_list.size(1),1)
            
            caption_att = self.softmax(self.att(caption_awear_q).squeeze(-1)) # batch_size num_cap
            
            caption_feat = torch.sum(caption_att.unsqueeze(-1) * caption_tensor_list, dim=-2) # batch_size module_dim
            
            v = question_embedding

            contextual_content = torch.cat([v,caption_feat],dim=-1) # batch_size module_dim*2

            #####################global-aggregation#####################
        
            global_embedding = self.Global_FeatureAggregation(video_motion_feat, video_object_feat, contextual_content) # (batch_size module*2)
            prospect_Background_embedding = self.Prospect_Background_aggregation(video_object_feat, video_appearance_feat, contextual_content) # (batch_size module*2)

            g_embedding = self.feature_aggregation_global(question_embedding, global_embedding)
            PB_embedding = self.feature_aggregation_PB(question_embedding, prospect_Background_embedding)


            out = self.fusion(torch.cat([g_embedding, PB_embedding, contextual_content], dim=-1))
            out = self.classifier(out)

        else:
            
            question_embedding = self.linguistic_input_unit(question, question_len) # batch_size module_dim

            # # corpus retrieve

            caption_tensor = corpus
            sort_, index_ = similarity.sort(1,descending=True) # batch_size num_cap
            
            index_ = index_[:,:self.topk].contiguous().view(-1, 1)  # batch_size*topk
            
            caption_tensor_list = caption_tensor[index_].contiguous().view(batch_size, self.topk, -1) # batch_size topk module_dim 
            
            #####################caption&question-attention#####################
            caption_awear_q = caption_tensor_list*question_embedding.unsqueeze(1).repeat(1,caption_tensor_list.size(1),1)
            caption_att = self.softmax(self.att(caption_awear_q).squeeze(-1)) # batch_size num_cap
            
            caption_feat = torch.sum(caption_att.unsqueeze(-1) * caption_tensor_list, dim=-2) # batch_size module_dim

            contextual_content = torch.cat([question_embedding,caption_feat], dim=-1) # batch_size module_dim*2

            #####################global-aggregation#####################
            
            global_embedding = self.Global_FeatureAggregation(video_motion_feat, video_object_feat, contextual_content) # (batch_size module*2)
            prospect_Background_embedding = self.Prospect_Background_aggregation(video_object_feat, video_appearance_feat, contextual_content) # (batch_size module*2)
            
            g_embedding = self.feature_aggregation_global(question_embedding, global_embedding)
            PB_embedding = self.feature_aggregation_PB(question_embedding, prospect_Background_embedding)
            
            out = self.fusion(torch.cat([g_embedding, PB_embedding, contextual_content], dim=-1))

            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_global_embedding = self.feature_aggregation_global(ans_candidates_embedding, global_embedding[batch_agg])
            a_PB_embedding = self.feature_aggregation_PB(ans_candidates_embedding, prospect_Background_embedding[batch_agg])

            a_visual_embedding = self.ffc(torch.cat([a_global_embedding, a_PB_embedding],dim=-1))

            out = self.output_unit(question_embedding[batch_agg], out[batch_agg],
                                    ans_candidates_embedding,
                                    a_visual_embedding, caption_feat[batch_agg])
        if self.visualization:
            return out, index_.reshape(batch_size,-1), caption_att

        return out


