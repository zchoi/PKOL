from email.policy import strict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import logging
import numpy as np
from tqdm import tqdm
import argparse
import sys
import json
import pickle
from termcolor import colored
from model.retrieve_model import RetrieveNetwork
from Dataloder_iterative import VideoQADataLoader
from utils import todevice

import model.PKOL as PKOL

from config import cfg, cfg_from_file
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

# msvd
# what 2 
# who 21
# how 54
# where 505
# when 405

# msrvtt
# what 10
# who 2
# how 64
# where 457
# when 310

question_type_acc_msvd = {2:0,21:0,54:0,505:0,405:0}
question_type_total_length_msvd = {2:0,21:0,54:0,505:0,405:0}

question_type_acc_msrvtt = {10:0,2:0,64:0,457:0,310:0}
question_type_total_length_msrvtt = {10:0,2:0,64:0,457:0,310:0}

def validate(cfg, model, retrieve_model, data, device, write_preds=False, test = False):
    model.eval()
    retrieve_model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    video_idx2_cap_gt = data.dataset.video_idx2_cap_gt

    caption_pool = data.dataset.caption_pool
    caption_pool_len = data.dataset.caption_pool_len

    model.topk = cfg.val.topk
    
    if test:
        if cfg.dataset.name == 'msvd-qa':
            with open('data/msvd-qa/ref_captions.json','r') as f:
                raw_caption = json.load(f)

    logging.info('top-k validation model: {}'.format(cfg.val.topk))
    logging.info("num of retrieve captions: {}".format(caption_pool.size(0)))
    logging.info('validation video length:{}'.format(len(video_idx2_cap_gt)))
    # model.corpus = caption_pool
    # model.corpus_len = caption_pool_len
    video_name = []
    all_scores = []
    caption_visualization = {}
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_idx, question_idx, answers, ans_candidates, ans_candidates_len, appearance_feat,\
                 motion_feat, object_feat, question, question_len = [todevice(x, device) for x in batch]
            # caption_pool = caption_pool.to(device)
            # caption_pool_len = caption_pool_len.to(device)
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            batch_size = motion_feat.size(0)

            with torch.no_grad():
                sim_list = []
                cap_list = []
                patch_num = cfg.train.patch_number    # 40000 -msrvtt  35000 -msvd
                chunk = data.dataset.caption_pool.size(0) // patch_num #1
                left = data.dataset.caption_pool.size(0) % patch_num   #22239
                j = 0
                for j in range(chunk):
                    cap = data.dataset.caption_pool[j*patch_num:(j+1)*patch_num].to(appearance_feat.device)
                    cap_len = data.dataset.caption_pool_len[j*patch_num:(j+1)*patch_num].to(appearance_feat.device)
                    similiry_j, caption_tensor_j = retrieve_model(   # batch_size patch_num / patch_num module_dim
                                    appearance_feat, 
                                    motion_feat, 
                                    cap, 
                                    cap_len,
                                    question,
                                    question_len
                                )
                    sim_list.append(similiry_j)
                    cap_list.append(caption_tensor_j)
                
                j = j+1 if chunk else j
                if left:
                    cap = data.dataset.caption_pool[j*patch_num:].to(appearance_feat.device)
                    cap_len = data.dataset.caption_pool_len[j*patch_num:].to(appearance_feat.device)
                    similiry_j, caption_tensor_j = retrieve_model(   # batch_size left / left module_dim
                                    appearance_feat, 
                                    motion_feat, 
                                    cap, 
                                    cap_len,
                                    question,
                                    question_len
                                )
                    
                    sim_list.append(similiry_j)
                    cap_list.append(caption_tensor_j)
                sim = torch.cat(sim_list, dim=-1)
                caption_tensor = torch.cat(cap_list, dim=0)            

            # sim, caption_tensor = retrieve_model(appearance_feat, motion_feat, caption_pool, caption_pool_len, question, question_len)
            if not test:
                logits = model(ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,
                    question_len, similarity=sim, corpus=caption_tensor)
            else:
                logits, index, caption_att = model(ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,
                    question_len, similarity=sim, corpus=caption_tensor)

            all_scores.append(sim.data.cpu().numpy())
            video_name.extend(list(video_idx.data.cpu().numpy()))

            if cfg.dataset.question_type in ['action', 'transition']:
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                agreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                preds = logits.detach().argmax(1)
                agreeings = (preds == answers)

            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count']:
                    preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition']:
                    answer_vocab = data.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        all_preds.append(predict.item())
                    else:
                        all_preds.append(answer_vocab[predict.item()])
                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])
                for id in video_idx:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_idx:
                    q_ids.append(ques_id.cpu().numpy())

            if cfg.dataset.question_type == 'count':
                total_acc += batch_mse.float().sum().item()
                count += answers.size(0)
            else:
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)
                if cfg.dataset.name == 'msvd-qa':
                    for h in range(question.size(0)):
                        if agreeings[h]:
                            question_type_acc_msvd[question[h][0].item()] += 1
                        question_type_total_length_msvd[question[h][0].item()] += 1
                elif cfg.dataset.name == 'msrvtt-qa':
                    for h in range(question.size(0)):
                        if agreeings[h]:
                            question_type_acc_msrvtt[question[h][0].item()] += 1
                        question_type_total_length_msrvtt[question[h][0].item()] += 1             
            if test:
                vocab = data.vocab['question_idx_to_token']
                answer_vocab = data.vocab['answer_idx_to_token']
                dict = {}
                with open(cfg.dataset.test_question_pt, 'rb') as f:
                    obj = pickle.load(f)
                    questions = obj['questions']
                    org_v_ids = obj['video_ids']
                    org_v_names = obj['video_names']
                    org_q_ids = obj['question_id']

                for idx in range(len(org_q_ids)):
                    dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
                for k, qid in enumerate(question_idx):
                    if answer_vocab[answers[k].item()] != answer_vocab[preds[k].item()]: #or answer_vocab[answers[k].item()]=='man' or answer_vocab[answers[k].item()]=='woman':
                        continue
                    for n, topk_i in enumerate(index[k]):
                        # caption_visualization.setdefault(qid.item(),[]).append((data.dataset.visualization[topk_i], video_idx[k].item()))
                        # if video_idx[k].item() != data.dataset.visualization[topk_i][0]:
                        #     continue
                        question = '' 
                        for word in dict[str(qid.item())][1]:
                            if word != 0:
                                question += vocab[word.item()] + ' '
                        
                        caption_visualization.setdefault(qid.item(),[]).append(
                            {   'caption': raw_caption['video'+str(video_idx[k].item())+'.mp4'][0],
                                'video_id': video_idx[k].item(), 
                                'retrieval_vid': data.dataset.visualization[topk_i][0],
                                'top-'+str(n)+'retrieval_cap': raw_caption['video'+str(data.dataset.visualization[topk_i][0])+'.mp4'][data.dataset.visualization[topk_i][1]],
                                'question':  question,
                                'answer': answer_vocab[answers[k].item()],
                                'prediction': answer_vocab[preds[k].item()]
                            })
        #############################################
        all_scores = np.concatenate(all_scores, axis= 0) # all_v all_c

        n_q, n_m = all_scores.shape
        gt_ranks = np.zeros((n_q, ), np.int32)

        for i in range(n_q):
            s = all_scores[i]
            sorted_idxs = np.argsort(-s)

            rank = n_m
            for k in video_idx2_cap_gt[str(video_name[i])]:
                tmp = np.where(sorted_idxs == k)[0][0]
                if tmp < rank:
                    rank = tmp
            gt_ranks[i] = rank
            
        r1 = 100 * len(np.where(gt_ranks < 1)[0]) / n_q
        r5 = 100 * len(np.where(gt_ranks < 5)[0]) / n_q
        r10 = 100 * len(np.where(gt_ranks < 10)[0]) / n_q
        # r1, r5, r10 = 0,0,0

        logging.info("r1: {:.4f}   r5: {:.4f}  r10:  {:.4f}".format(r1,r5,r10))
        #############################################
        
        acc = total_acc / count
    if test:
        output_dir = os.path.join(cfg.dataset.save_dir,'visualization')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "visualization.json")
        with open(preds_file,'w') as f:
            json.dump(caption_visualization, f)
# what 2 
# who 21
# how 54
# where 505
# when 405
# question_type_acc = {2:0,21:0,54:0,505:0,405:0}
    if cfg.dataset.name == 'msvd-qa':
        logging.info("What:{:.4f} Who: {:.4f} How: {:.4f} When: {:.4f} Where: {:.4f}".format(
                                question_type_acc_msvd[2]/question_type_total_length_msvd[2],  
                                question_type_acc_msvd[21]/question_type_total_length_msvd[21], 
                                question_type_acc_msvd[54]/question_type_total_length_msvd[54],  
                                question_type_acc_msvd[405]/question_type_total_length_msvd[405], 
                                question_type_acc_msvd[505]/question_type_total_length_msvd[505]                                                  
                                        ))
    elif cfg.dataset.name == 'msrvtt-qa':
        logging.info("What:{:.4f} Who: {:.4f} How: {:.4f} When: {:.4f} Where: {:.4f}".format(
                        question_type_acc_msrvtt[10]/question_type_total_length_msrvtt[10],  
                        question_type_acc_msrvtt[2]/question_type_total_length_msrvtt[2], 
                        question_type_acc_msrvtt[64]/question_type_total_length_msrvtt[64],  
                        question_type_acc_msrvtt[310]/question_type_total_length_msrvtt[310], 
                        question_type_acc_msrvtt[457]/question_type_total_length_msrvtt[457]                                                  
                                ))  
    if not write_preds:
        return acc, r1, r5, r10
    else:
        return acc, all_preds, gts, v_ids, q_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/tgif_qa_action.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name, cfg.dataset.pretrained)
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    ckpt_retrieval = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model_retrieval.pt')

    assert os.path.exists(ckpt) and os.path.exists(ckpt_retrieval)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    loaded_retrieval = torch.load(ckpt_retrieval, map_location='cpu')
    model_kwargs_retrieval = loaded_retrieval['model_kwargs']

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.object_feat = '/mnt/hdd2/zhanghaonan/object_features.h5'
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_feat.h5'
        cfg.dataset.object_feat = '{}_object_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.test_question_pt = '{}_test_questions.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.object_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.object_feat.format(cfg.dataset.name))
        
    test_loader_kwargs = {
            'split' : 'test',
            'name' : cfg.dataset.name,
            'caption_max_num' : cfg.dataset.max_cap_num,
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.test_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'object_feat' : cfg.dataset.object_feat,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
            
        }
    
    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model_kwargs.update({'visualization': cfg.test.visualization})
    model = PKOL.PKOL_Net(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'], strict=False)

    model_kwargs_retrieval.update({'vocab': test_loader.vocab})
    retrieve_model = RetrieveNetwork(**model_kwargs_retrieval).to(device)
    retrieve_model.load_state_dict(loaded_retrieval['state_dict'], strict=False)

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, retrieve_model, test_loader, device, write_preds=True, test=cfg.test.visualization)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_idx']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': dict[str(q_id)][0], 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 samples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 examples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc, _, _, _ = validate(cfg, model, retrieve_model, test_loader, device, write_preds=False, test=cfg.test.visualization)
        # acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
