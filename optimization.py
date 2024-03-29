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
    
    return feature, vid_len, sample_idx
def process_label( vid_len, sample_idx):
    vid_duration, vid_fps = gt_dict[vid_name]['duration'], gt_dict[vid_name]['fps']

    if NUM_SEGMENTS == -1:
        t_factor_point = args.frames_per_sec / (vid_fps * 16)
        temp_anno = np.zeros([vid_len, num_class], dtype=np.float32)
        temp_df = point_anno[point_anno["video_id"] == vid_name][['point', 'class']]
        for key in temp_df['point'].keys():
            point = temp_df['point'][key]
            class_idx = class_idx_dict[temp_df['class'][key]]
            temp_anno[int(point * t_factor_point)][class_idx] = 1
        point_label = temp_anno[sample_idx, :]
        return  point_label, vid_duration
    
    else:
        t_factor_point = NUM_SEGMENTS / (vid_fps * vid_duration)
        temp_anno = np.zeros([NUM_SEGMENTS, num_class], dtype=np.float32)
        temp_df = point_anno[point_anno["video_id"] == vid_name][['point', 'class']]
        for key in temp_df['point'].keys():
            point = temp_df['point'][key]
            class_idx = class_idx_dict[temp_df['class'][key]]
            temp_anno[int(point * t_factor_point)][class_idx] = 1
        point_label = temp_anno
        return vid_label, point_label, vid_duration

def create_sample(file_path):

    vid_name= file_path.split('/')[-1][:-4]
    #TODO: video feature 추출해서 바로 인풋에 넣는거 연결해야됨
    vid_feature = np.load(os.path.join(file_path))
    
    data, vid_len, sample_idx = process_feat(vid_feature)
    fps=30
    vid_duration = vid_len* 16/ fps
    # vid_duration = process_label(vid_name, vid_len, sample_idx)
    data= np.expand_dims(data,axis=0)
    data =torch.from_numpy(data).float().to(args.device)
    sample = dict(
        data = data, 
        vid_name = vid_name, 
        vid_len = vid_len, 
        vid_duration = vid_duration,
    )
    return sample
@torch.no_grad()
def S_test(net, args, sample):
    net.eval()
    snippet_result = {}
    snippet_result['version'] = 'VERSION 1.3'
    snippet_result['results'] = {}
    snippet_result['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

    num_correct = 0.
    num_total = 0.
    
    _data,_vid_len, vid_duration = sample['data'],  sample['vid_len'] ,sample['vid_duration']

    # _vid_len =torch.tensor(_vid_len)
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
        proposals = utils.get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments, vid_duration)
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
        proposals = utils.get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments, vid_duration)
        for i in range(len(proposals)):
            class_id = proposals[i][0][2]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[i]

    final_proposals = post_process(args, proposal_dict)

    return  final_proposals
    # json_path = os.path.join(args.output_path_s1, 'snippet_result_{}.json'.format(subset, args.seed))
    # with open(json_path, 'w') as f:
    #     json.dump(snippet_result, f, cls=NumpyArrayEncoder)
         
    # if args.mode == 'train' or args.mode == 'infer':
    #     test_acc = num_correct / num_total
    #     print("TEST ACC:{:.4f}".format(test_acc))
    #     test_map = log_evaluate(args, step, test_acc, logger, json_path, test_info, subset)
    #     return test_map
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
    final_result = {}
    #TODO:sample 채우기
    features, proposals = sample['data'], sample['proposals']

    # features = [torch.from_numpy(feat).float().to(args.device) for feat in features]
    proposals_input = [torch.from_numpy(prop).float().to(args.device) for prop in proposals]
    outputs = net(features, proposals_input, is_training=False)

    proposal_dict = get_prediction(proposals[0], outputs)
    final_proposals = post_process(args , proposal_dict)

    final_result['results'] = final_proposals

    return final_result

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


def main(args):
    # >> Initialize the task
    # save_config(args, os.path.join(args.output_path_s1, "config.json"))
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
    
    sample=create_sample("/home/bigdeal/mnt2/HR-Pro/dataset/THUMOS14/features/test/video_test_0001409.npy")
    start_time=time.time()
    stage1_proposals =S_test(model, args, sample)
    PP_proposals =reliability_ranking(args,stage1_proposals)
    stage2_data=dict(
    data = sample["data"],
    proposals = load_proposals(PP_proposals),
    )
    final_proposal = I_test( args, stage2_data, model2)
    print(final_proposal)    
    end_time=time.time()
    print(end_time-start_time)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)