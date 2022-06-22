import json
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
import model.PKOL as PKOL


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


from Dataloder_iterative import VideoQADataLoader
from utils import todevice
from validate_iterative import validate
from model.retrieve_model import RetrieveNetwork
from utils import todevice
from termcolor import colored
from config import cfg, cfg_from_file


def margin_ranking_loss(
        similary_matrix, 
        margin=None, 
        direction= 'both', 
        average_batch = True, 
        video_name = None,
        video_idx2_cap_gt=None
        ):

    batch_size = similary_matrix.size(0)
    diagonal = similary_matrix.diag().view(batch_size, 1)

    pos_mask = torch.eye(batch_size,batch_size,device=similary_matrix.device).bool()

    # v2c
    if direction == 'both' or direction == 'v2c':
        diagonal_1 = diagonal.expand_as(similary_matrix)
        
        cost_cap = (margin + similary_matrix - diagonal_1).clamp(min=0)
        cost_cap = cost_cap.masked_fill(pos_mask, 0)
        if average_batch:
            cost_cap = cost_cap / (batch_size * (batch_size - 1))
            cost_cap = torch.sum(cost_cap)

    # c2v
    if direction == 'both' or direction == 'c2v':
        diagonal_2 = diagonal.t().expand_as(similary_matrix)
        cost_vid = (margin + similary_matrix - diagonal_2).clamp(min=0)
        cost_vid = cost_vid.masked_fill(pos_mask,0)
        if average_batch:
            cost_vid = cost_vid / (batch_size * (batch_size - 1))
            cost_vid = torch.sum(cost_vid)
    
    if direction == 'both':
        return cost_cap + cost_vid
    elif direction == 'v2c':
        return cost_cap
    else:
        return cost_vid

def train(cfg):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'split' : 'train',
        'name' : cfg.dataset.name,
        'caption_max_num' : cfg.dataset.max_cap_num,
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.train_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'object_feat' : cfg.dataset.object_feat,
        'train_num': cfg.train.train_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True,
        'pin_memory': True
        
    }
    train_loader = VideoQADataLoader(**train_loader_kwargs)

    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'split' : 'val',
            'name' : cfg.dataset.name,
            'caption_max_num' : cfg.dataset.max_cap_num,
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'object_feat' : cfg.dataset.object_feat,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'pin_memory': True
            
        }
        val_loader = VideoQADataLoader(**val_loader_kwargs)

        logging.info("number of val instances: {}".format(len(val_loader.dataset)))
    
    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('data/tgif-qa/tgif-caption/tgif_cap_index.json','r') as f:
        f = json.load(f)
    tgif_cap = f if cfg.dataset.question_type != 'none' else None
    
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'k_max_frame_level': cfg.train.k_max_frame_level,
        'k_max_clip_level': cfg.train.k_max_clip_level, 
        'spl_resolution': cfg.train.spl_resolution,
        'vocab': train_loader.vocab,
        'question_type': cfg.dataset.question_type,
        'caption_dim' : cfg.train.caption_dim,
        'topk' : cfg.train.topk,
        'corpus' : None,
        'corpus_len' : None,
        'patch_number' : cfg.train.patch_number,
        'cap_vocab' : tgif_cap if cfg.dataset.question_type != 'none' else None,
        'visualization' : False
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    model = PKOL.PKOL_Net(**model_kwargs).to(device)
    
    retrieve_model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'k_max_frame_level': cfg.train.k_max_frame_level,
        'k_max_clip_level': cfg.train.k_max_clip_level, 
        'spl_resolution': cfg.train.spl_resolution,
        'vocab': train_loader.vocab,
        'question_type': cfg.dataset.question_type,
        'caption_dim' : cfg.train.caption_dim,
        'cap_vocab' : tgif_cap if cfg.dataset.question_type != 'none' else None 
    }
    model_retrieval_kwargs_tosave = {k: v for k, v in retrieve_model_kwargs.items() if k != 'vocab'}
    retrieve_model = RetrieveNetwork(**retrieve_model_kwargs).to(device)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pytorch_total_params_R = sum(p.numel() for p in retrieve_model.parameters() if p.requires_grad)

    logging.info('top-k trained model: {}'.format(cfg.train.topk))
    logging.info('num of params: {}'.format(pytorch_total_params + pytorch_total_params_R))
    #logging.info(model)

    if cfg.train.glove:
        logging.info('load glove vectors')
        train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
        with torch.no_grad():
            model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
            retrieve_model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
            if cfg.dataset.question_type != 'none':
                retrieve_model.word_embedding.data = torch.from_numpy(np.load('data/tgif-qa/tgif-caption/glove.npy')).to(device)
                
    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=None)

    optimizer = optim.Adam([{'params': model.parameters()},{'params': retrieve_model.parameters()}], cfg.train.lr)

    start_epoch = 0

    if cfg.dataset.question_type == 'count':
        best_val = 100.0
    else:
        best_val = 0
        best_retrieval = 0
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

        ckpt_retrieval = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model_retrieval.pt')   
        retrieve_model.load_state_dict(ckpt_retrieval['state_dict'])

    if cfg.dataset.question_type in ['frameqa', 'none']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.dataset.question_type == 'count':
        criterion = nn.MSELoss().to(device)

    logging.info("Start training........")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {} <<<<<<'.format(epoch))
        if epoch < 0:
            retrieve_model.train()
            count = 0
            batch_mse_sum = 0.0
            total_loss, avg_loss = 0.0, 0.0
            avg_loss = 0
            
            for i, batch in enumerate(iter(train_loader)):
                progress = epoch + i / len(train_loader)
                video_idx, question_idx, answers, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, _, question,\
                    question_len, caption, caption_len = [todevice(x, device) for x in batch]
                batch_size = appearance_feat.size(0)
                optimizer.zero_grad()
                sim_matrix, _ = retrieve_model(appearance_feat, motion_feat, caption, caption_len, question, question_len)
                
                loss = margin_ranking_loss(sim_matrix,0.2)
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(retrieve_model.parameters(), max_norm=12)
                optimizer.step()

                sys.stdout.write(
                    "\rProgress = {progress}  avg_loss = {avg_loss}  exp: {exp_name}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        exp_name=cfg.exp_name))

                sys.stdout.flush()
            sys.stdout.write("\n")

            if (epoch + 1) % 10 == 0:
                optimizer = step_decay(cfg, optimizer)
            sys.stdout.flush()
            torch.cuda.empty_cache()
        else:
            model.train()
            retrieve_model.train()
            total_acc, count = 0, 0
            batch_mse_sum = 0.0
            total_loss, avg_loss = 0.0, 0.0
            avg_loss = 0
            train_accuracy = 0
            for i, batch in enumerate(iter(train_loader)):
                progress = epoch + i / len(train_loader)
                video_idx, question_idx, answers, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,\
                    question_len, caption, caption_len = [todevice(x, device) for x in batch]

                answers = answers.cuda().squeeze()
                batch_size = answers.size(0)
                optimizer.zero_grad()

                sim, _ = retrieve_model(appearance_feat, motion_feat, caption, caption_len, question, question_len)
                
                with torch.no_grad():
                    sim_list = []
                    cap_list = []
                    patch_num = cfg.train.patch_number    # 40000 -msrvtt  35000 -msvd
                    chunk = train_loader.dataset.caption_pool.size(0) // patch_num #1
                    left = train_loader.dataset.caption_pool.size(0) % patch_num   #22239
                    j = 0
                    for j in range(chunk):
                        cap = train_loader.dataset.caption_pool[j*patch_num:(j+1)*patch_num].to(appearance_feat.device)
                        cap_len = train_loader.dataset.caption_pool_len[j*patch_num:(j+1)*patch_num].to(appearance_feat.device)
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
                        cap = train_loader.dataset.caption_pool[j*patch_num:].to(appearance_feat.device)
                        cap_len = train_loader.dataset.caption_pool_len[j*patch_num:].to(appearance_feat.device)
                        similiry_j, caption_tensor_j = retrieve_model(   # batch_size left / left module_dim
                                        appearance_feat, 
                                        motion_feat, 
                                        cap, 
                                        cap_len,
                                        question,
                                        question_len)
                        
                        sim_list.append(similiry_j)
                        cap_list.append(caption_tensor_j)
                    similiry_matrix = torch.cat(sim_list, dim=-1)
                    caption_tensor = torch.cat(cap_list, dim=0)

                logits = model(ans_candidates, ans_candidates_len, appearance_feat, motion_feat, object_feat, question,
                question_len, similarity=similiry_matrix, corpus=caption_tensor)  # batch_size batch_size
                
                if cfg.dataset.question_type in ['action', 'transition']:   
                    batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                                    [1, 5])) * 5  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
                    answers_agg = tile(answers, 0, 5)
                    loss = torch.max(torch.tensor(0.0).cuda(),
                                    1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
                    loss = loss.sum()
                    if cfg.train.joint:
                        r_loss = margin_ranking_loss(
                            similary_matrix = sim, 
                            margin=0.2
                            )
                        loss += r_loss
                    loss.backward()
                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                    aggreeings = (preds == answers)
                elif cfg.dataset.question_type == 'count':
                    answers = answers.unsqueeze(-1)
                    loss = criterion(logits, answers.float())
                    if cfg.train.joint:
                        r_loss = margin_ranking_loss(
                            similary_matrix = sim, 
                            margin=0.2
                            )
                        loss += r_loss
                    loss.backward()
                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    preds = (logits + 0.5).long().clamp(min=1, max=10)
                    batch_mse = (preds - answers) ** 2
                else:
                    loss = criterion(logits, answers)
                    if cfg.train.joint:
                        r_loss = margin_ranking_loss(
                            similary_matrix = sim, 
                            margin=0.2
                            )
                        loss += r_loss 
                    loss.backward()
                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    aggreeings = batch_accuracy(logits, answers)

                if cfg.dataset.question_type == 'count':
                    batch_avg_mse = batch_mse.sum().item() / answers.size(0)
                    batch_mse_sum += batch_mse.sum().item()
                    count += answers.size(0)
                    avg_mse = batch_mse_sum / count
                    sys.stdout.write(
                        "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}    exp: {exp_name}".format(
                            progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                            ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                            avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                            train_mse=colored("{:.4f}".format(batch_avg_mse), "blue",
                                            attrs=['bold']),
                            avg_mse=colored("{:.4f}".format(avg_mse), "red", attrs=['bold']),
                            exp_name=cfg.exp_name))
                    sys.stdout.flush()
                else:
                    total_acc += aggreeings.sum().item()
                    count += answers.size(0)
                    train_accuracy = total_acc / count
                    if not cfg.train.joint:
                        sys.stdout.write(
                            "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}    exp: {exp_name}".format(
                                progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                                ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                                avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                                train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                                attrs=['bold']),
                                avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                                exp_name=cfg.exp_name))
                    else:
                        sys.stdout.write(
                            "\rProgress = {progress}   ce_loss = {ce_loss}   re_loss = {re_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}    exp: {exp_name}".format(
                                progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                                ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                                re_loss=colored("{:.4f}".format(r_loss.item()), "blue", attrs=['bold']),
                                avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                                train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                                attrs=['bold']),
                                avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                                exp_name=cfg.exp_name))
                    sys.stdout.flush()
            
            sys.stdout.write("\n")
            if cfg.dataset.question_type == 'count':
                if (epoch + 1) % 5 == 0:
                    optimizer = step_decay(cfg, optimizer)
            else:
                if (epoch + 1) % 10 == 0:
                    optimizer = step_decay(cfg, optimizer)
            sys.stdout.flush()
            
            logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))
            if cfg.val.flag:
                output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    assert os.path.isdir(output_dir)
                valid_acc, _, _, r10 = validate(cfg, model, retrieve_model, val_loader, device, write_preds=False)
                if (valid_acc > best_val and cfg.dataset.question_type != 'count') or (valid_acc < best_val and cfg.dataset.question_type == 'count'):
                    best_val = valid_acc
                    # Save best model
                    ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    else:
                        assert os.path.isdir(ckpt_dir)
                    save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model.pt'))
                    save_checkpoint(epoch, retrieve_model, optimizer, model_retrieval_kwargs_tosave, os.path.join(ckpt_dir, 'model_retrieval.pt'))
                    sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                    sys.stdout.flush()
                
                logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
                sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
                sys.stdout.flush()

# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='msvd_qa.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    # check if k_max is set correctly
    assert cfg.train.k_max_frame_level <= 16
    assert cfg.train.k_max_clip_level <= 8


    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)

    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")

    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
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
        cfg.dataset.train_question_pt = '{}_train_questions.pt'
        cfg.dataset.val_question_pt = '{}_val_questions.pt'
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.object_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.object_feat.format(cfg.dataset.name))
        
    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train(cfg)


if __name__ == '__main__':
    main()
