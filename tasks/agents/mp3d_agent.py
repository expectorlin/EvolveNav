import math
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.ops import pad_tensors_wgrad
from models.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from .base_agent import BaseAgent

import torch
from collections import defaultdict
from contextlib import nullcontext
from models.graph_utils import GraphMap
from typing import List

import json
import os
import random

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)

    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks

def get_results(pred_results, detailed_output=False):
    pred_output = []
    for k, v in pred_results.items():
        ret = {
            'instr_id': k,
            'trajectory': v['path']
        }
        # scan_qa
        if 'answer' in v:
            ret.update({
                'pred_answer': v['generated_sentences'],
                'oracle_pred_answer': v.get('oracle_pred_answer', ''),
                'gt_answer': v['answer'],
            })
        
        # obj nav
        if 'pred_objid' in v:
            ret.update({
                'pred_objid': v['pred_objid'],
                'pred_obj_direction': v['pred_obj_direction']
            })
        pred_output.append(ret)

    return pred_output

def remove_article(item):
    if item[:2] == 'a ' or item[:3] == 'an ' or item[:4] == 'the ':
        item = ' '.join(item.split(' ')[1:])
    return item

class MP3DAgent(BaseAgent):
    def __init__(self, args, shortest_distances, shortest_paths):
        self.args = args
        self.shortest_paths = shortest_paths
        self.shortest_distances = shortest_distances
        # buffer
        self.scanvp_cands = {}
        self.t2t_landmark_dir = str(args.data_dir / 't2t_landmarks')

    def update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def panorama_feature_variable_object(self, obs):
        ''' Extract precomputed features into variable. '''
        has_obj = 'obj_img_fts' in obs[0]

        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens, batch_obj_loc_fts = [], [], []
        batch_cand_vpids, batch_objids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = torch.from_numpy(np.concatenate([view_ang_fts, view_box_fts], 1))

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))
            batch_loc_fts.append(view_loc_fts)

            # object
            if has_obj:
                batch_obj_loc_fts.append(torch.from_numpy(np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)))
                batch_objids.append(ob['obj_ids'])
                batch_obj_lens.append(len(ob['obj_img_fts']))
                batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        ret = {
            'view_img_fts': batch_view_img_fts,
            'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types,
            'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids, 
        }

        if has_obj:
            batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
            batch_obj_loc_fts = pad_tensors(batch_obj_loc_fts).cuda()
            batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()
            assert batch_obj_img_fts.shape[:2] == batch_obj_loc_fts.shape[:2], f'shape of batch_obj_img_fts {batch_obj_img_fts.shape[:2]} must equal to shape of batch_obj_loc_fts {batch_obj_loc_fts.shape[:2]}'
            ret.update({
                'obj_img_fts': batch_obj_img_fts,
                'obj_loc_fts': batch_obj_loc_fts,
                'obj_lens': batch_obj_lens,
                'obj_ids': batch_objids,
            })

        return ret
    
    def panorama_feature_variable_12views(self, obs):
        batch_view_img_fts = []
        batch_loc_fts = []
        batch_view_lens = []
        batch_nav_types = []
        batch_cand_vpids = []

        for i, ob in enumerate(obs):
            view_img_fts = [x[:self.args.image_feat_size]  for k,x in enumerate(ob['feature'])]
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = [x[self.args.image_feat_size:] for k, x in enumerate(ob['feature'])]
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)  
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_view_lens.append(len(view_img_fts))
            batch_nav_types.append(torch.LongTensor([1]*12+[0]*24))
            batch_cand_vpids.append([None]*36)

        
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        ret = {
            "view_img_fts": batch_view_img_fts,
            "loc_fts": batch_loc_fts,
            "nav_types": batch_nav_types,
            "view_lens": batch_view_lens,
            "cand_vpids":  batch_cand_vpids
        }
        return ret

    def get_pos_fts(self, cnt_vp, cand_vps, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in cand_vps:
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=cur_elevation,
            )
            rel_angles.append([rel_heading, rel_elevation])
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return rel_ang_fts

    def prepare_cot(self, obs, nav_vpids, nav_targets, gmaps, landmark_direction=None):
        batch_navigation_cot_gt = []
        direction_of_gt_list = []

        for i, ob in enumerate(obs):
            if nav_targets[i] == -100:
                nav_target = 0
            else:
                nav_target = nav_targets[i]

            gt_vpid = nav_vpids[i][nav_target]
            if gt_vpid is not None:

                direction_of_gt = self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']], gmaps[i].node_positions[gt_vpid], obs[i]['heading'], obs[i]['elevation'], use_mapping=True)
                direction_of_gt_list.append(direction_of_gt)
                action_of_gt = ' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']], gmaps[i].node_positions[gt_vpid], obs[i]['heading'], obs[i]['elevation']).split(' ')[:-1])

                landmarks_of_gt = landmark_direction[i][action_of_gt]['landmarks']

                cot_label = f"I should go to an observation with [{', '.join(landmarks_of_gt)}] {direction_of_gt} me."
            else:
                direction_of_gt = None
                direction_of_gt_list.append(direction_of_gt)
                #print('gt_vpid is None!!!!!!')
                cot_label = f"Observation matches with long-term goal, so stop."


            cot_label = f"{cot_label}"

            navigation_cot_gt = cot_label

            batch_navigation_cot_gt.append(navigation_cot_gt)

        return batch_navigation_cot_gt, direction_of_gt_list

    def prepare_landmark_direction(self, obs,  nav_vpids, gmaps,  vp_landmarks, cand_masks):

        batch_summary = []
        for i, ob in enumerate(obs):
            cur_nav_vpids = [nav_vpids[i][x] for x in range(len(nav_vpids[i])) if cand_masks[i][x]]

            summary = {}
            for j, vp in enumerate(cur_nav_vpids):
                if j > 0 :
                    vp_direction =' '.join(self.get_direction_vp(gmaps[i].node_positions[obs[i]['viewpoint']],
                                                      gmaps[i].node_positions[vp], obs[i]['heading'], obs[i]['elevation']).split(' ')[:2]) # delete 'to' in direction

                    if vp_direction not in summary:
                        summary[vp_direction] = {}
                        summary[vp_direction]['landmarks'] = set()
                        summary[vp_direction]['cand_index'] = []
                    summary[vp_direction]['landmarks'].update(vp_landmarks[i][vp])
                    summary[vp_direction]['cand_index'].append(j)

            directions = list(summary.keys())
            for direction in directions:
                if len(summary[direction]['landmarks']) > self.args.land_num:
                    summary[direction]['landmarks'] = random.sample(summary[direction]['landmarks'], self.args.land_num)

            batch_summary.append(summary)
        return batch_summary


    def get_direction_vp(self, cnt_vp, vp, cur_heading, use_mapping=False):
        rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=0
            )

        if use_mapping:

            action_direction_mapping = {
                'go forward to': "in front of",
                'go back to': "behind",
                'turn right to': "to the right of",
                'turn left to': "to the left of",
                'go up to': "above",
                'go down to': "below"
            }

        rel_elevation = math.degrees(rel_elevation)

        if rel_elevation > 10:
            direction_text = 'go up to'
        elif rel_elevation < -10:
            direction_text = 'go down to'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 4:
                    direction_text = 'go forward to'
                elif rel_heading < -math.pi / 4 and rel_heading >= -math.pi * 3 / 4:
                    direction_text = 'turn left to'
                elif rel_heading < -math.pi * 3 / 4 and rel_heading >= -math.pi * 5 / 4:
                    direction_text = 'go back to'
                elif rel_heading < -math.pi * 5 / 4 and rel_heading >= -math.pi * 7 / 4:
                    direction_text = 'turn right to'
                else:
                    direction_text = 'go forward to'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 4:
                    direction_text = 'go forward to'
                elif rel_heading > math.pi / 4 and rel_heading <= math.pi * 3 / 4:
                    direction_text = 'turn right to'
                elif rel_heading > math.pi * 3 / 4 and rel_heading <= math.pi * 5 / 4:
                    direction_text = 'go back to'
                elif rel_heading > math.pi * 5 / 4 and rel_heading <= math.pi * 7 / 4:
                    direction_text = 'turn left to'
                else:
                    direction_text = 'go forward to'
            elif rel_heading == 0:
                direction_text = 'go forward to'

        if use_mapping:
            direction_text = action_direction_mapping[direction_text]

        return direction_text


    def get_pos_fts(self, cnt_vp, cand_vps, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in cand_vps:
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=cur_elevation,
            )
            rel_angles.append([rel_heading, rel_elevation])
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return rel_ang_fts

    def nav_vp_variable(self, obs, gmaps, pano_embeds, pano_masks, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )
        pano_masks = torch.cat(
            [torch.ones_like(pano_masks[:, :1]), pano_masks], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'pano_masks': pano_masks,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }


    def nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                is_r2r = 'r2r' in ob['instr_id']
                if imitation_learning and is_r2r:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:

                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    pass
                                    # dist = - cal_dtw(
                                    #     self.shortest_distances[scan],
                                    #     sum(traj[i]['path'], []) + self.shortest_paths[scan][ob['viewpoint']][vpid][1:],
                                    #     ob['gt_path'],
                                    #     threshold=3.0
                                    # )['nDTW']
                                elif self.args.expert_policy == 'spl':

                                    dist = self.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()


    def teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()


    def teacher_object(self, obs):
        targets = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            i_vp = ob['viewpoint']
            i_objids = ob['obj_ids']
            if len(i_objids) == 0:
                targets[i] = self.args.ignoreid
            else:
                targets[i] = self.args.ignoreid      # target is not exist among the candidates
                if i_vp in ob['gt_end_vps']:
                    for j, obj_id in enumerate(i_objids):
                        if str(obj_id) == str(ob['gt_obj_id']):
                            targets[i] = j + 1
                            break
        return torch.from_numpy(targets).cuda()


    def make_equiv_action(self, a_t, gmaps, obs, traj=None, env=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                env[i].sims[0].newEpisode([ob['scan']], [action], [heading], [elevation])



    def train(
        self, 
        name,
        batch,
        args,
        config,
        model,
        criterion,
        dataset,
        step=0,
        entropy_metric=None,
        instr_pred_metric=None,
        **kwargs
    ):
        dataset_cfg = config.Pretrain if args.stage=='pretrain' else config.Finetune
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        if args.stage=='pretrain' or step%2==0  or args.IL_setting:
            #################### imitation learning ####################
            loss, _ = self.rollout(
                args, name, config.Optim, batch,
                model=model, criterion=criterion, dataset=dataset,
                feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
            )

        else:
            #################### dagger training ####################
            loss, _ = self.rollout(
                args, name, config.Optim, batch,
                model=model, criterion=criterion, dataset=dataset,
                feedback="sample", train_ml=loss_coef,
                entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
            )

        return loss * args.gradient_accumulation_step


    def validate(
        self,
        name,
        args,
        config,
        model,
        loader,
        entropy_metric=None,
        instr_pred_metric=None,
    ):
        results = {}
        trie = None
        looped = False
        dataset = loader.get_dataset()
        pbar = tqdm(loader, disable=args.rank!=0)
        if name in ['EQA']:
            if hasattr(model, 'module'):
                tokenizer = model.module.lang_model.tokenizer
            else:
                tokenizer = model.lang_model.tokenizer

            trie = Trie(tokenizer.bos_token_id, tokenizer.eos_token_id)
            for word in dataset.answer_vocab:
                token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                if isinstance(tokenizer, LlamaTokenizer):
                    token_ids = [tokenizer.bos_token_id] + token_ids
                trie.insert(token_ids)

        for i, batch in enumerate(pbar):
            ml_loss, traj = self.rollout(
                args, name, config.Optim, batch,
                model=model, criterion=None, dataset=dataset,
                feedback= "sample" if args.do_sample else "argmax", train_ml=None,
                entropy_metric=entropy_metric, instr_pred_metric=None,
                validate=True, trie=trie
            )

            for s_traj in traj:
                if s_traj['instr_id'] in results:
                    looped = True
                else:
                    ml_loss = 0
                    results[s_traj['instr_id']] = s_traj
    
            # Caldulate oracle prediction answer
            if name in ["EQA"]:
                _, oracle_traj = self.rollout(
                    args, name, config.Optim, batch,
                    model=model, criterion=None, dataset=dataset,
                    feedback="teacher", train_ml=1,
                    entropy_metric=entropy_metric, instr_pred_metric=None,
                    validate=True, trie=trie
                )
                for s_traj in oracle_traj:
                    results[s_traj['instr_id']]['oracle_pred_answer'] = s_traj['generated_sentences']

            if looped:
                break
        
        preds = get_results(results)
        return preds

    def rollout(
        self,
        args,
        name,
        config,
        batch_dict,
        model,
        criterion,
        dataset,
        feedback,
        train_ml,
        entropy_metric,
        instr_pred_metric,
        validate=False,
        **kwargs
    ):
        """
        :param args:
        :param name: task name
        :param config:
        :param batch_dict:
        :param model:
        :param criterion:
        :param dataset:
        :param feedback:
        :param train_ml:
        :param entropy_metric:
        :param validate:
        :return:
        """
        obs = batch_dict["observations"]
        envs = batch_dict["env"]
        data_type = batch_dict['data_type']

        max_action_len = config.val_max_action_len[name] if validate else config.train_max_action_len[name]

        self.update_scanvp_cands(obs)
        batch_size = len(obs)

        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # instructions = language_variable(obs, data_type=batch_dict['data_type'])
        instructions = [ob["instruction"] for ob in obs]

        history = []
        hist_vis = []
        for idx in range(len(instructions)):
            history.append([])
            hist_vis.append([])

        entropys = []
        ml_loss, cnt_loss = 0., 0.
        flag = False

        vp_landmarks = [{} for ob in obs]

        for t in range(max_action_len):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # multi-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                else:
                    context = model.no_sync
            else:
                # single-gpu
                if ended.all() or t == max_action_len - 1:
                    flag = True
                    context = nullcontext
                else:
                    context = nullcontext

            with context():
                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

                # graph representation
                pano_inputs = self.panorama_feature_variable_object(obs)
                panorama_out = model('panorama', pano_inputs)
                pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        # update visited node
                        i_vp = obs[i]['viewpoint']
                        update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                        gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                        # update unvisited nodes
                        for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                            if not gmap.graph.visited(i_cand_vp):
                                update_pano_embeds = pano_embeds[i, j].detach()
                                gmap.update_node_embed(i_cand_vp, update_pano_embeds)

                # navigation policy
                nav_inputs = self.nav_gmap_variable(obs, gmaps)
                nav_inputs.update(
                    self.nav_vp_variable(
                        obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                    )
                )

                nav_inputs.update({
                    'view_lens': pano_inputs['view_lens'],
                    'instruction': instructions,
                    'history': history,
                    'hist_vis': hist_vis,
                    'data_type': data_type
                })

                for i, ob in enumerate(obs):
                    for j, cc in enumerate(ob['candidate']):
                        if cc['viewpointId'] not in vp_landmarks[i]:
                            vp_landmarks[i][cc['viewpointId']]=[remove_article(item) for item in ob['landmarks'][f"{cc['pointId']}"]]
                nav_inputs.update({
                    'vp_landmarks': vp_landmarks
                })

                in_progress = torch.tensor(ended).logical_not()
                if ended.all():
                    in_progress[0] = True

                nav_vpids = nav_inputs['gmap_vpids']
                imitation_learning = feedback == 'teacher'


                if 'r2r' in data_type:
                    nav_targets = self.teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'],
                        imitation_learning=imitation_learning, t=t, traj=traj
                    )
                else:
                    nav_targets = self.teacher_action(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'],
                    )


                cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())

                nav_inputs.update({
                "landmark_direction": self.prepare_landmark_direction(obs,
                                                      nav_vpids=nav_vpids, nav_targets=nav_targets,
                                                      gmaps=gmaps, vp_landmarks=vp_landmarks,cand_masks=cand_masks
                                                      )})

                navigation_cot_gt, direction_of_gt = self.prepare_cot(obs, nav_vpids=nav_vpids, nav_targets=nav_targets,
                                                                       gmaps=gmaps,
                                                                       landmark_direction=nav_inputs['landmark_direction'] if 'landmark_direction' in nav_inputs else None)

                nav_inputs.update({
                    "navigation_cot_gt": navigation_cot_gt,
                "direction_of_gt":direction_of_gt})


                nav_inputs["prompts"] = self.prepare_prompts(
                    "navigation_cot_and_decision",
                    nav_inputs,
                    cls_token=model.module.lang_model.cls_token[0] if hasattr(model, 'module') else
                    model.lang_model.cls_token[0]
                )


                if self.args.Stage_2_training and not validate:
                    with torch.no_grad():
                        output_generate = model('navigation_cot_and_decision', nav_inputs,
                                            training=False, cot_generate=True,
                                            **kwargs)

                    batch_generated_sentences_navigation_cot = output_generate[
                            "generated_sentences_navigation_cot"]

                    nav_logits = output_generate['fuse_logits']
                    new_cot_gt = []
                    if train_ml is not None:
                        _, a_t = nav_logits.max(1)
                        for i in range(len(navigation_cot_gt)):
                            if a_t[i] == nav_targets[i]:
                                if random.random() < self.args.replace_cot_gt_prob:
                                    new_cot_gt_i = batch_generated_sentences_navigation_cot[i]
                                    new_cot_gt_i = new_cot_gt_i.replace('<s>','')
                                    new_cot_gt_i = new_cot_gt_i.replace('</s>','')
                                    # avoid bad output causing out of memory
                                    if int(len(new_cot_gt_i)) < int(2 * len(navigation_cot_gt[i])) and len(new_cot_gt_i) > 5:
                                        new_cot_gt.append(new_cot_gt_i)
                                    else:
                                        new_cot_gt.append(navigation_cot_gt[i])
                                else:
                                    new_cot_gt.append(navigation_cot_gt[i])
                            else:
                                new_cot_gt.append(navigation_cot_gt[i])

                    nav_inputs.update({
                        "navigation_cot_gt": new_cot_gt})

                    direction_landmark_dict = output_generate["direction_landmark_dict"]
                    rand_perms = output_generate["rand_perms"]

                output = model('navigation_cot_and_decision', nav_inputs, training=not validate, **kwargs)
                if not validate:
                    lm_loss = output["loss"]  / batch_size / args.gradient_accumulation_step

                # dynamic fusion
                nav_logits = output['fuse_logits']
                nav_vpids = nav_inputs['gmap_vpids']

                nav_probs = torch.softmax(nav_logits / args.temperature, 1)

                imitation_learning = feedback == 'teacher'
                # Imitation Learning
                if train_ml is not None:
                    # [1] Supervised training
                    if 'r2r' in data_type:
                        nav_targets = self.teacher_action_r4r(
                            obs, nav_vpids, ended,
                            visited_masks=nav_inputs['gmap_visited_masks'],
                            imitation_learning=imitation_learning, t=t, traj=traj
                        )
                    else:
                        nav_targets = self.teacher_action(
                            obs, nav_vpids, ended,
                            visited_masks=nav_inputs['gmap_visited_masks'],
                    )
                    ############# Single-Step Loss #############
                    cnt_loss += criterion(nav_logits, nav_targets) * train_ml / batch_size / args.gradient_accumulation_step

                    step_wise_cal_lmloss_prob = random.random()

                    if step_wise_cal_lmloss_prob < self.args.cal_lmloss_prob:
                        #print("calculate lm loss!!!")
                        if not flag:
                            cnt_loss += lm_loss
                        print(
                            f"step{t}: lm_loss:{lm_loss.detach()}, cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")
                    else:
                        # lm_loss.detach()
                        print(
                            f"step{t}: cnt_loss: {cnt_loss.detach()}, ml_loss: {ml_loss}")


                    ml_loss += cnt_loss.detach()

                    ########### Single-Step Backward ###########
                    if not validate:
                        cnt_loss.backward()
                    cnt_loss = 0.

                # Determinate the next navigation viewpoint
                if feedback == 'teacher':  # imitation learning
                    a_t = nav_targets  # teacher forcing
                elif feedback == 'sample':
                    c = torch.distributions.Categorical(nav_probs.float())
                    entropy_metric.accumulate(c.entropy().sum().item()/ batch_size)  # For log
                    entropys.append(c.entropy())  # For optimization
                    a_t = c.sample().detach()
                elif feedback == 'argmax':
                    _, a_t = nav_logits.max(1)  # student forcing - argmax
                    a_t = a_t.detach()
                else:
                    print(feedback)
                    raise NotImplementedError

                for idx in range(len(a_t)):
                    if a_t[idx] == -100:
                        continue
                    history[idx] += ['<hist>']
                    fuse_embeds = output['fuse_embeds']
                    hist_vis[idx].append(fuse_embeds[idx][a_t[idx]])

                if not validate:
                    # if feedback == 'teacher' or feedback == 'sample':  # in training
                    assert feedback in ['teacher', 'sample'], "Feedback must be either `teacher' or `sample' in training. "
                    a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
                else:
                    a_t_stop = a_t == 0

                ########### Self-Reflective Sub-task ###########
                enable_self_reflective = args.Stage_2_training and not validate
                if enable_self_reflective:
                    pano_inputs = self.panorama_feature_variable_object(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']

                    # navigation policy
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    nav_inputs.update({
                        'vp_landmarks': vp_landmarks
                    })

                    nav_inputs.update({
                        'view_lens': pano_inputs['view_lens'],
                        'instruction': instructions,
                        'history': history,
                        'hist_vis': hist_vis,
                        'data_type': data_type
                    })

                    cand_masks = torch.clone(nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not())

                    nav_inputs.update({
                    "landmark_direction": self.prepare_landmark_direction(obs,
                                                          nav_vpids=nav_vpids, nav_targets=nav_targets,
                                                          gmaps=gmaps, vp_landmarks=vp_landmarks,cand_masks=cand_masks
                                                          )})

                    nav_inputs.update({
                        "navigation_cot_gt": new_cot_gt})


                    pos_cot = nav_inputs['navigation_cot_gt']

                    neg_cot = self.get_neg_cot(nav_inputs['navigation_cot_gt'], direction_landmark_dict, direction_of_gt)

                    nav_inputs.update({
                        "navigation_self_reflective_pos": pos_cot,
                        "navigation_self_reflective_neg": neg_cot
                    })

                    batch_cot_pair = []
                    batch_self_reflective_gt = []
                    for i in range(batch_size):
                        order = list(range(2))
                        random.shuffle(order)
                        ori_pair = [nav_inputs['navigation_self_reflective_pos'][i], nav_inputs["navigation_self_reflective_neg"][i]]
                        new_pair = [ori_pair[x] for x in order]
                        self_reflective_gt = f'Output {order.index(0)+1}.' # pos_cot is at index 0 in ori_pair
                        batch_cot_pair.append(new_pair)
                        batch_self_reflective_gt.append(self_reflective_gt)

                    nav_inputs.update({
                        'navigation_cot_pair': batch_cot_pair,
                        "navigation_cot_gt": batch_self_reflective_gt
                    })

                    nav_inputs["prompts"] = self.prepare_prompts(
                        "navigation_self_reflective",
                        nav_inputs
                    )

                    self_reflective_output = model('navigation_cot', nav_inputs, training=not validate, rand_perms=rand_perms,
                                   **kwargs)

                    if not validate:
                        self_reflective_lm_loss = self_reflective_output["loss"] * args.self_reflective_loss_weight / batch_size / args.gradient_accumulation_step

                        self_reflective_lm_loss.backward()

                        instr_pred_metric.accumulate(self_reflective_lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += self_reflective_lm_loss.detach()
                        print(f"step{t}: self_reflective_loss:{self_reflective_lm_loss.detach()}")


                ########### Fine-grained R2R Sub-task ###########
                enable_fgr2r = (feedback == 'teacher') and (not flag) and (not a_t_stop[0]) and (data_type[0]=='r2r') and (not validate) and 'fg_instruction' in ob and args.enable_fgr2r
                if enable_fgr2r:
                    pano_inputs = self.panorama_feature_variable_12views(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    nav_inputs['instruction'] = ['where are we going with direction ({}) ?'.format(idx) for idx in nav_targets]
                    nav_inputs["data_type"] = ['fgr2r' for idx in nav_targets]
                    nav_inputs['answer'] = [ob['fg_instruction'][ob['fg_view'][t]] for ob in obs]
                    nav_inputs['hist_vis'] = [[] for idx in nav_targets]
                    nav_inputs['history'] = [[] for idx in nav_targets]
                    nav_inputs["prompts"] = self.prepare_prompts("embodied_qa", nav_inputs)
                    output = model('embodied_qa', nav_inputs, training=not validate, **kwargs)
                    if not validate:
                        lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        lm_loss.backward()
                        instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += lm_loss.detach()
            
                ########### Navigation Summarization Sub-task ###########
                if data_type[0] == 'eqa':
                    enable_summarize = flag
                elif data_type[0] in ['r2r']:
                    enable_summarize = (feedback == 'teacher' or feedback == 'argmax') and flag and args.enable_summarize and (not validate or args.mode=='test')
                elif data_type[0] in ['cvdn']:
                    enable_summarize = False
                else:
                    raise NotImplementedError

                if enable_summarize:  # gen loss
                    pano_inputs = self.panorama_feature_variable_12views(obs)
                    panorama_out = model('panorama', pano_inputs)
                    pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                    nav_inputs = self.nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self.nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    
                    nav_inputs['instruction'] = [ob["instruction"] for ob in obs]
                    nav_inputs['history'] = history
                    nav_inputs['hist_vis'] = hist_vis
                    nav_inputs["data_type"] = data_type
                    nav_inputs['answer'] = [ob.get('answer', '') for ob in obs]
                    nav_inputs["prompts"] = self.prepare_prompts("summarization", nav_inputs)
                    output = model('summarization', nav_inputs, training=not validate, **kwargs)
                    if not validate:
                        lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                        lm_loss.backward()
                        instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                        ml_loss += lm_loss.detach()

                # Prepare environment action
                cpu_a_t = []
                for i in range(batch_size):
                    # TODO
                    if False and data_type[i] == 'eqa':
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == max_action_len - 1):
                            cpu_a_t.append(None)
                            just_ended[i] = True
                        else:
                            cpu_a_t.append(nav_vpids[i][a_t[i]])

                # Make action and get the new state
                self.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

                for i in range(batch_size):
                    if (not ended[i]) and just_ended[i]:
                        stop_node, stop_score = None, {'stop': -float('inf')}
                        for k, v in gmaps[i].node_stop_scores.items():
                            if v['stop'] > stop_score['stop']:
                                stop_score = v
                                stop_node = k
                        if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                            traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

                # get new observation and update graph
                new_obs = []
                for b_i in range(batch_size):
                    # TODO
                    if False and data_type[b_i] == 'eqa':
                        raise NotImplementedError
                    else:
                        new_obs.append(
                            dataset.get_obs(
                                items=[batch_dict['item'][b_i]],
                                env=envs[b_i], data_type=data_type[b_i]
                            )[0]
                        )
                obs = new_obs

                self.update_scanvp_cands(obs)

                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmaps[i].update_graph(ob)

                ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

                if flag:
                    break

        return ml_loss, traj

    def get_neg_cot(self, gt_cot, direction_landmark_dict, direction_of_gt):

        bs = len(gt_cot)
        neg_cot = []

        direction_gt_mapping = {
            'go forward':"in front of",
            'go back':"behind",
            'turn right':"to the right of",
            'turn left':"to the left of",
            'go up':"above",
            'go down':"below"
        }

        for bs_ind in range(bs):

            direction_keys = list(direction_landmark_dict[bs_ind].keys())
            if direction_of_gt[bs_ind] not in direction_keys:
                replace_direction = random.choice(direction_keys)
                replace_landmark = direction_landmark_dict[bs_ind][replace_direction]
                modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."

                neg_cot.append(modify_output_cot)
            else:
                direction_keys.remove(direction_of_gt[bs_ind])
                replace_direction = random.choice(direction_keys)
                replace_landmark = direction_landmark_dict[bs_ind][replace_direction]
                modify_output_cot = f"I should go to an observation with {replace_landmark} {direction_gt_mapping[replace_direction]} me."

                neg_cot.append(modify_output_cot)

        return neg_cot

    def prepare_prompts(self, mode, batch, **kwargs):
        batch_size = len(batch["instruction"])
        if mode == "navigation":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                    )
                )
        elif mode == "summarization" or mode == "embodied_qa":
            hist_nums = [len(his) for his in batch["history"]]
            vp_nav_masks = batch["vp_nav_masks"][:, 1:]
            cand_nums = vp_nav_masks.sum(1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        mode,
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                    )
                )
        elif mode == "object_grounding":
            hist_nums = [len(his) for his in batch["history"]]
            cand_nums = batch["obj_masks"].sum(dim=1) + 1    # add not exist
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        mode,
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                    )
                )

        elif mode == "navigation_cot_and_decision":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_cot_and_decision",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cls_token=kwargs.get("cls_token"),
                    )
                )

        elif mode == "navigation_self_reflective":
            hist_nums = [len(his) for his in batch["history"]]
            cand_masks = torch.clone(batch['gmap_masks'] & batch['gmap_visited_masks'].logical_not())
            cand_nums = cand_masks.sum(dim=-1)
            prompts = []
            for bn in range(batch_size):
                prompts.append(
                    self.get_prompt(
                        "navigation_self_reflective",
                        instruction=batch["instruction"][bn],
                        hist_num=hist_nums[bn],
                        cand_num=cand_nums[bn],
                        cot_pair=batch['navigation_cot_pair'][bn]
                    )
                )

        else:
            raise NotImplementedError

        return prompts