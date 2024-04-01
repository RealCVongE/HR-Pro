import argparse
import os
import torch
import torch.utils.data as data
from tqdm import tqdm
from tensorboard_logger import Logger
import time
import datetime
import numpy as np
import yaml
import options
import utils
from dataset import dataset
from model_S import S_Model
from model_I import I_Model
from train import S_train, I_train
from test import NumpyArrayEncoder, S_test, I_test
# from log import log_evaluate, save_config, initial_log, save_best_record
from ranking import reliability_ranking
import json
NUM_SEGMENTS=-1
SAMPLE="random"

def init_args(args):
    # create folder for models/outputs/logs of stage1/stage2
    args.root_s1 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage1')
    args.model_path_s1 = os.path.join(args.root_s1, 'models' )
    args.output_path_s1 = os.path.join(args.root_s1, "outputs")
    args.log_path_s1 = os.path.join(args.root_s1, "logs")

    args.root_s2 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage2')
    args.model_path_s2 = os.path.join(args.root_s2, 'models' )
    args.output_path_s2 = os.path.join(args.root_s2, "outputs")
    args.log_path_s2 = os.path.join(args.root_s2, "logs")

    for dir in [args.model_path_s1, args.log_path_s1, args.output_path_s1,
                args.model_path_s2, args.log_path_s2, args.output_path_s2]:
        options.mkdir(dir)

    # mapping parameters of string format
    args.act_thresh_cas = eval(args.act_thresh_cas)
    args.act_thresh_agnostic = eval(args.act_thresh_agnostic)
    args.lambdas = eval(args.lambdas)
    args.tIoU_thresh = eval(args.tIoU_thresh)
    
    # get list of class name 
    args.class_name_lst = options._CLASS_NAME[args.dataset]
    args.num_class = len(args.class_name_lst)

    # define format of test information
    args.test_info = {
        "step": [], "test_acc": [], 'loss':[], 'elapsed': [], 'now': [],
        "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
        "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [], 
        "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []
    }
    return args

def parse_args():
    parser = argparse.ArgumentParser("Official Pytorch Implementation of HR-Pro: Point-supervised Temporal Action Localization \
                                        via Hierarchical Reliability Propagation")
    
    parser.add_argument('--cfg', type=str, default='thumos', help='hyperparameters path')
    parser.add_argument('--seed', type=int, default=0, help='random seed (-1 for no manual seed)')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='root folder for saving models,ouputs,logs')
    args = parser.parse_args()

    # hyper-params from ymal file
    with open('./cfgs/{}_hyp.yaml'.format(args.cfg)) as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in hyp_dict.items():
        setattr(args, key, value)

    return init_args(args)
def load_proposals(proposals):
    proposals_new = None
        # >> caculate t_factor(time --> snippet)
    t_factor =  args.frames_per_sec / args.segment_frames_num
    class_idx_dict = {cls: idx for idx, cls in enumerate(args.class_name_lst)}
    # >> load Positive Proposals
    PP = proposals
    PP_new = []
    for label in PP.keys():
        for prop in PP[label]:
            prop_new = [round(prop[0]*t_factor, 3), round(prop[1]*t_factor, 3), prop[2], class_idx_dict[label]]
            PP_new.append(prop_new)
    if len(PP_new) == 0:
        PP_new.append([0, 0, 0])
    proposals_new = np.array(PP_new)
    return np.expand_dims(proposals_new,axis=0) 

       

def post_process(args, proposal_dict ):
    final_proposals = []
    for class_id in proposal_dict.keys():
        temp_proposal = utils.soft_nms(proposal_dict[class_id], sigma=0.3)
        final_proposals += temp_proposal
    final_proposals = utils.result2json(args, final_proposals)

    return final_proposals
def process_feat( vid_feature):
    vid_len = vid_feature.shape[0]
    if vid_len <= NUM_SEGMENTS or NUM_SEGMENTS == -1:
        sample_idx = np.arange(vid_len).astype(int)
    elif NUM_SEGMENTS > 0 and SAMPLE == "random":
        sample_idx = np.arange(NUM_SEGMENTS) * vid_len / NUM_SEGMENTS
        for i in range(NUM_SEGMENTS):
            if i < NUM_SEGMENTS - 1:
                if int(sample_idx[i]) != int(sample_idx[i + 1]):
                    sample_idx[i] = np.random.choice(range(int(sample_idx[i]), int(sample_idx[i + 1]) + 1))
                else:
                    sample_idx[i] = int(sample_idx[i])
            else:
                if int(sample_idx[i]) < vid_len - 1:
                    sample_idx[i] = np.random.choice(range(int(sample_idx[i]), vid_len))
                else:
                    sample_idx[i] = int(sample_idx[i])
    elif NUM_SEGMENTS > 0 and SAMPLE == 'uniform':
        samples = np.arange(NUM_SEGMENTS) * vid_len / NUM_SEGMENTS
        samples = np.floor(samples)
        sample_idx =  samples.astype(int)
    else:
        raise AssertionError('Not supported sampling !')
    feature = vid_feature[sample_idx]
    
    return feature, vid_len

def create_sample(file_path):
    vid_name= file_path.split('/')[-1][:-4]
    vid_feature = np.load(os.path.join(file_path))
    data, vid_len = process_feat(vid_feature)
    data= np.expand_dims(data,axis=0)
    data =torch.from_numpy(data).float().to(args.device)
    sample = dict(
        data = data, 
        vid_name = vid_name, 
        vid_len = vid_len, 
    )
    return sample
@torch.no_grad()
def S_test(net, args, sample):
    net.eval()
    _data,_vid_len= sample['data'],  sample['vid_len']
    b=0
    outputs = net(_data.to(args.device))
    _vid_score, _cas_fuse = outputs['vid_score'], outputs['cas_fuse']
    vid_len = _vid_len
    # >> caculate video-level prediction
    score_np = _vid_score[b].cpu().numpy()
    pred_np = np.zeros_like(score_np)
    pred_np[np.where(score_np < args.class_thresh)] = 0
    pred_np[np.where(score_np >= args.class_thresh)] = 1
    if pred_np.sum() == 0:
        pred_np[np.argmax(score_np)] = 1
    # >> post-process
    cas_fuse = _cas_fuse[b]
    num_segments = _data[b].shape[0]
    # class-specific score
    cas_S = cas_fuse[:, :-1]
    pred = np.where(score_np >= args.class_thresh)[0]
    if len(pred) == 0:
        pred = np.array([np.argmax(score_np)])
    cas_pred = cas_S.cpu().numpy()[:, pred]   
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
    cas_pred = utils.upgrade_resolution(cas_pred, args.scale)
    # class-agnostic score
    agnostic_score = 1 - cas_fuse[:, -1].unsqueeze(1)
    agnostic_score = agnostic_score.expand((-1, args.num_class))
    agnostic_score = agnostic_score.cpu().numpy()[:, pred]
    agnostic_score = np.reshape(agnostic_score, (num_segments, -1, 1))
    agnostic_score = utils.upgrade_resolution(agnostic_score, args.scale)
    # >> generate proposals
    proposal_dict = {}
    for i in range(len(args.act_thresh_cas)):
        cas_temp = cas_pred.copy()
        zero_location = np.where(cas_temp[:, :, 0] < args.act_thresh_cas[i])
        cas_temp[zero_location] = 0

        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp[:, c, 0] > 0)
            seg_list.append(pos)
        proposals = get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][2]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[i]

    for i in range(len(args.act_thresh_agnostic)):
        cas_temp = cas_pred.copy()
        agnostic_score_temp = agnostic_score.copy()
        zero_location = np.where(agnostic_score_temp[:, :, 0] < args.act_thresh_agnostic[i])
        agnostic_score_temp[zero_location] = 0
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(agnostic_score_temp[:, c, 0] > 0)
            seg_list.append(pos)
        proposals =get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][2]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[i]
    final_proposals = post_process(args, proposal_dict)

    return  final_proposals
def get_proposal_oic(args, tList, wtcam, vid_score, c_pred, v_len, num_segments):
    t_factor = float(16 * v_len) / ( args.scale * num_segments * args.frames_per_sec )
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = utils.grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - args._lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + args._lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + \
                                    list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + args.gamma * vid_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([t_start, t_end, c_pred[i], c_score])
            temp.append(c_temp)
    return temp

def get_prediction(proposals, data_dict):
    t_factor =  args.frames_per_sec / args.segment_frames_num
    proposal_dict = {}
    prop_iou = data_dict['iou_pred_orig'][0].cpu().numpy()
    for i in range(proposals.shape[0]):
        c = int(proposals[i,3])
        if c not in proposal_dict.keys():
            proposal_dict[c] = []
        c_score = prop_iou[i, 0] + proposals[i, 2]
        proposal_dict[c].append([proposals[i, 0] / t_factor, proposals[i, 1] / t_factor, c, c_score])

    prop_iou = data_dict['iou_pred_refined'][0].cpu().numpy()
    proposals = data_dict['prop_refined'][0].cpu().numpy()
    for i in range(proposals.shape[0]):
        c = int(proposals[i,3])
        if c not in proposal_dict.keys():
            proposal_dict[c]=[]
        c_score = prop_iou[i, 0] + proposals[i, 2]
        proposal_dict[c].append([proposals[i, 0] / t_factor, proposals[i, 1] / t_factor, c, c_score])

    return  proposal_dict

@torch.no_grad()
def I_test( args, sample, net):
    net.eval()
    features, proposals = sample['data'], sample['proposals']
    proposals_input = [torch.from_numpy(prop).float().to(args.device) for prop in proposals]
    outputs = net(features, proposals_input, is_training=False)
    proposal_dict = get_prediction(proposals[0], outputs)
    final_proposals = post_process(args , proposal_dict)
    return final_proposals

def reliability_ranking(args, stage1_proposals):
    '''
    point-based proposal generation
    '''
    # proposals_dict = dict()
    # for subset in ['test', 'train']:
    #     snippet_result_path = os.path.join(args.output_path_s1, 'snippet_result_{}.json'.format(subset))
    #     assert os.path.exists(snippet_result_path)
    #     with open(snippet_result_path, 'r') as json_file:
    #         snippet_result = json.load(json_file)

    sub_proposals_dict = dict()
        # >> Positive Proposals
    PP = {}
    for pred in stage1_proposals:
        label, segment, score = pred['label'], pred['segment'], pred['score']
        if label not in PP.keys():
            PP[label] = []
        PP[label].append([segment[0], segment[1], score])
    sub_proposals_dict = PP
    return sub_proposals_dict


def hr_pro(args):
    utils.set_seed(args.seed)
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # --------------------------------------------------Snippet-level Optimization-------------------------------------------------------#
    model = S_Model(args)
    model = model.to(args.device)
        # --------------------------------------------------Instance-level Optimization-------------------------------------------------------#
    model2 = I_Model(args)
    model2 = model2.to(args.device)

    model.load_state_dict(torch.load(os.path.join(args.model_path_s1, "model1_seed_{}.pkl".format(args.seed))))
    model2.load_state_dict(torch.load(os.path.join(args.model_path_s2, "model2_seed_{}.pkl".format(args.seed))))

    sample=create_sample("video4.npy")
    start_time=time.time()
    stage1_proposals =S_test(model, args, sample)
    PP_proposals =reliability_ranking(args,stage1_proposals)
    stage2_data=dict(
    data = sample["data"],
    proposals = load_proposals(PP_proposals),
    )
    final_proposal = I_test( args, stage2_data, model2)
    # threshold = 0.
    # filtered_data = [item for item in final_proposal if item['score'] > threshold]
    end_time=time.time()

    print(final_proposal)    
    print(end_time-start_time)
    
if __name__ == "__main__":
    args = parse_args()
    hr_pro(args)