import numpy as np
from torch.nn import functional as F
from torch.nn.modules import module
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import *

class DynamicRNN(nn.Module):
  def __init__(self, rnn_model):
    super().__init__()
    self.rnn_model = rnn_model

  def forward(self, seq_input, seq_lens, initial_state=None):
    """A wrapper over pytorch's rnn to handle sequences of variable length.

    Arguments
    ---------
    seq_input : torch.Tensor
        Input sequence tensor (padded) for RNN model.
        Shape: (batch_size, max_sequence_length, embed_size)
    seq_lens : torch.LongTensor
        Length of sequences (b, )
    initial_state : torch.Tensor
        Initial (hidden, cell) states of RNN model.

    Returns
    -------
        Single tensor of shape (batch_size, rnn_hidden_size) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence.
    """
    max_sequence_length = seq_input.size(1)
    sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
    sorted_seq_input = seq_input.index_select(0, fwd_order)
    packed_seq_input = pack_padded_sequence(
      sorted_seq_input, lengths=sorted_len, batch_first=True
    )

    if initial_state is not None:
      hx = initial_state
      assert hx[0].size(0) == self.rnn_model.num_layers
    else:
      sorted_hx = None

    self.rnn_model.flatten_parameters()

    outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, sorted_hx)

    # pick hidden and cell states of last layer
    h_n = h_n[-1].index_select(dim=0, index=bwd_order)
    c_n = c_n[-1].index_select(dim=0, index=bwd_order)

    outputs = pad_packed_sequence(
      outputs, batch_first=True, total_length=max_sequence_length
    )[0].index_select(dim=0, index=bwd_order)

    return outputs, (h_n, c_n)

  @staticmethod
  def _get_sorted_order(lens):
    sorted_len, fwd_order = torch.sort(
      lens.contiguous().view(-1), 0, descending=True
    )
    _, bwd_order = torch.sort(fwd_order)
    sorted_len = list(sorted_len)
    return sorted_len, fwd_order, bwd_order

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


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
        # self.encoder = DynamicRNN(self.encoder)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.caption_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, caption, caption_len):
        """
        Args:
            caption: [Tensor] (batch_size, max_question_length, cap_dim)
            caption_len: [Tensor] (batch_size,)
        return:
            caption representation [Tensor] (batch_size, module_dim)
        """
        bs, max_len, rnn_dim = caption.size()
        #caption_embedding = self.encoder_embed(caption)  # (batch_size, num_cap, max_question_length, rnn_dim)
        caption_embedding = caption
        embed = self.tanh(self.embedding_dropout(caption_embedding))
        
        #for i in range(num_cap):
            
        embed_candi = nn.utils.rnn.pack_padded_sequence(embed, caption_len.cpu(), batch_first=True,
                                                enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (caption_embedding, _) = self.encoder(embed_candi)
        if self.bidirectional:
            caption_embedding = torch.cat([caption_embedding[0], caption_embedding[1]], -1)
        caption_embedding = self.caption_dropout(caption_embedding)
        
        return caption_embedding


class RetrieveNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type, caption_dim, cap_vocab):
        super(RetrieveNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
        ########################################
        if cap_vocab is not None:
            self.word_embedding = nn.Embedding(len(cap_vocab), word_dim)
        
        self.caption_unit = captionLinguistic(caption_dim=caption_dim)
        self.appearance_W = nn.Sequential(
            nn.Linear(vision_dim,module_dim),
            nn.Dropout(p=0.1),
            nn.Tanh()
        )
        self.appearance_ATT = nn.Sequential(
            nn.Linear(module_dim,1),
            nn.Softmax(dim=-1)
        )
        self.motion_W = nn.Sequential(
            nn.Linear(vision_dim,module_dim),
            nn.Dropout(p=0.1),
            nn.Tanh()
        )
        self.motion_ATT = nn.Sequential(
            nn.Linear(module_dim,1),
            nn.Softmax(dim=-1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(module_dim*4,module_dim*2),
            nn.Dropout(p=0.2),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)
        ########################################
        init_modules(self.modules(), w_init="xavier_uniform")
        if cap_vocab is not None:
            nn.init.uniform_(self.word_embedding.weight, -1.0, 1.0)

    def forward(self, video_appearance_feat, video_motion_feat, caption, caption_len, question=None, question_len=None):
        """
        Args:
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            caption: [Tensor] (batch_size/num_cap num_word cap_dim) | [tgif] (batch_size/num_cap num_word)
            caption_len: [Tensor] (batch_size/num_cap) | [tgif] (batch_size/num_cap)
            question: [Tensor] (batch_size num_word)
            question_len [Tensor] (batch_size) 
        return:
            similarity matrix.
        """

        batch_size = video_appearance_feat.size(0)
        _,_,num_frames,_ = video_appearance_feat.size()
        # video -> caption
        caption_embedding = caption

        if self.question_type != 'none':
            
            caption_embedding = self.word_embedding(caption) # batch_size num_word embedding
        
        caption_sentence = self.caption_unit(caption_embedding,caption_len)  # batch_size module_dim
        
        appearance_feat = self.appearance_W(video_appearance_feat) # batch_size num_clips num_frames module_dim
        
        motion_feat = self.motion_W(video_motion_feat) # batch_size num_K module_dim
        
        #appearance_feat = torch.sum(motion_feat.unsqueeze(-2).repeat(1,1,num_frames,1)*appearance_feat,dim=-2) # batch_size num_K module_dim
        appearance_feat = torch.mean(appearance_feat,dim=-2).squeeze() # batch_size num_clips module_dim
        appearance_ATT = self.appearance_ATT(appearance_feat)  # batch_size num_K 1
        motion_ATT = self.motion_ATT(motion_feat)  # batch_size num_K 1
        
        appearance = torch.bmm(appearance_ATT.permute(0,2,1),appearance_feat).squeeze(-2) # batch_size module_dim  
        motion = torch.bmm(motion_ATT.permute(0,2,1),motion_feat).squeeze(-2) # batch_size module_dim

        v = ((appearance + motion)/2)
        inner_prod = v.mm(caption_sentence.t()) # video cap
        
        im_norm = torch.sqrt((v**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((caption_sentence**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        # question -> caption
        
        if question is not None:
            
            question_embedding = self.linguistic_input_unit(question, question_len) # (batch_size, module_dim)
            q_c_sim = question_embedding.mm(caption_sentence.t())  # ques cap
            que_norm_ = torch.sqrt((q_c_sim**2).sum(1).view(-1, 1) + 1e-18)
            q_c_sim = q_c_sim / (que_norm_ * s_norm)
            
        return sim + q_c_sim, caption_sentence
        # return sim, caption_sentence
        
class ContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=False, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)
      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)
      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im