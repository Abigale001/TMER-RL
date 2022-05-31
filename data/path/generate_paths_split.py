# from data.path.rl_model import SMCP
# from data.path.rl_model import GaussianPolicy
# from data.path.rl_model import ReplayMemory

import sys
sys.path.append('../../')
import argparse
import random
import numpy as np
import time
import pickle
import torch
from collections import defaultdict
from itertools import chain
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from random import sample

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
# from rl_model import *
from data.path.rl_model import *

import math


class Prepared_data:
    def __init__(self, **kargs):
        self.ui_dict = dict()
        self.iu_dict = dict()
        self.ic_dict = dict()
        self.ci_dict = dict()
        self.ib_dict = dict()
        self.bi_dict = dict()
        self.usize = kargs.get('usize')
        self.isize = kargs.get('isize')
        self.csize = kargs.get('csize')
        self.bsize = kargs.get('bsize')
        self.embeddings = np.zeros((self.usize + self.isize + self.csize + self.bsize, 100))
        self.user_embedding = np.zeros((self.usize, 100))
        self.item_embedding = np.zeros((self.isize, 100))
        self.category_embedding = np.zeros((self.csize, 100))
        self.brand_embedding = np.zeros((self.bsize, 100))

        self.userid = torch.arange(0, self.usize)
        self.itemid = torch.arange(self.usize, self.usize + self.isize)
        self.categoryid = torch.arange(self.usize + self.isize, self.usize + self.isize + self.csize)
        self.brandid = torch.arange(self.usize + self.isize + self.csize,
                                    self.usize + self.isize + self.csize + self.bsize)

        print('Begin to load data')
        start = time.time()

        self.load_embedding(kargs.get('node_emb_dic'))

        self.load_ui(kargs.get('ui_relation_file'))
        self.load_ic(kargs.get('ic_relation_file'))
        self.load_ib(kargs.get('ib_relation_file'))

        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))

    ###123

    def load_embedding(self, embfile):
        nodewv_dic = pickle.load(open(embfile, 'rb'))
        nodewv_tensor = []
        all_nodes = list(range(len(nodewv_dic.keys())))
        for node in all_nodes:
            nodewv_tensor.append(nodewv_dic[node].numpy())
        nodewv_tensor = torch.Tensor(nodewv_tensor)
        self.embeddings = nodewv_tensor
        # because when we save all embeddings, the order is user, item, category and brand
        self.user_embedding = nodewv_tensor[:self.usize, :]
        self.item_embedding = nodewv_tensor[self.usize : self.usize + self.isize, :]
        self.category_embedding = nodewv_tensor[self.usize + self.isize : self.usize + self.isize + self.csize, :]
        self.brand_embedding = nodewv_tensor[self.usize + self.isize + self.csize : self.usize + self.isize + self.csize + self.bsize, :]

        return self.user_embedding, self.item_embedding, self.category_embedding, self.brand_embedding

    def load_ui(self, uifile):
        user_item_data = open(uifile, 'r').readlines()
        for user_item_ele in user_item_data:
            user_item_ele_list = user_item_ele.strip().split(',')
            user = int(user_item_ele_list[0])
            item = int(user_item_ele_list[1])
            if item not in self.iu_dict.keys():
                self.iu_dict[item] = [user]
            else:
                self.iu_dict[item].append(user)

            if user not in self.ui_dict.keys():
                self.ui_dict[user] = [item]
            else:
                self.ui_dict[user].append(item)
        return self.ui_dict, self.iu_dict

    def load_ib(self, ibfile):
        item_brand_data = open(ibfile, 'r').readlines()
        for item_brand_ele in item_brand_data:
            item_brand_ele_list = item_brand_ele.strip().split(',')
            item = int(item_brand_ele_list[0])
            brand = int(item_brand_ele_list[1])
            if item not in self.ib_dict.keys():
                self.ib_dict[item] = [brand]
            else:
                self.ib_dict[item].append(brand)

            if brand not in self.bi_dict.keys():
                self.bi_dict[brand] = [item]
            else:
                self.bi_dict[brand].append(item)
        return self.ib_dict, self.bi_dict

    def load_ic(self, icfile):
        item_category_data = open(icfile, 'r').readlines()
        for item_category_ele in item_category_data:
            item_category_ele_list = item_category_ele.strip().split(',')
            item = int(item_category_ele_list[0])
            category = int(item_category_ele_list[1])
            if item not in self.ic_dict.keys():
                self.ic_dict[item] = [category]
            else:
                self.ic_dict[item].append(category)

            if category not in self.ci_dict.keys():
                self.ci_dict[category] = [item]
            else:
                self.ci_dict[category].append(item)
        return self.ic_dict, self.ci_dict


# candidate neighbours 数量不固定
def find_candidate(start, data, no_neighobor=[]):

    # The strategy to explore user-item paths and item-item paths is the same.

    userid = data.userid
    itemid = data.itemid
    categoryid = data.categoryid
    brandid = data.brandid
    candidate_neighbors = []
    try:
        if start in userid:
            candidate_neighbors = data.ui_dict[start]

        elif start in itemid:
            candidate_neighbors = data.ic_dict[start] + data.ib_dict[start]

        elif start in categoryid:
            candidate_neighbors = data.ci_dict[start]

        elif start in brandid:
            candidate_neighbors = data.bi_dict[start]

        if len(candidate_neighbors) < 50:
            candidate_neighbors_without_noneighbor = [item for item in candidate_neighbors if item not in no_neighobor]
            return candidate_neighbors_without_noneighbor
        else:
            return candidate_neighbors
    except KeyError:
        return -1


# todo: want to delete no neighbor nodes
def top_k(start, candidate_neighbors, data):
    refinefolder = args.refinefolder
    ic_relation_file = refinefolder + 'item_category.relation'
    ib_relation_file = refinefolder + 'item_brand.relation'
    ui_relation_file = refinefolder + 'user_item.relation'
    node_emb_dic = args.basedatafolder + 'nodewv.dic'
    user_history_file = args.userhistoryfile
    usize = args.usize
    isize = args.isize
    csize = args.csize
    bsize = args.bsize


    h1 = data.embeddings[start]
    h1 = torch.Tensor(h1)

    ##candidate_id 和 neighbour_id是一样的
    action_values = []
    candidate_id = []
    # print(f'candidate_neighrbor:{candidate_neighbors}')

    if len(candidate_neighbors) <= 50:
        # if this neighbor doesn't have a neighbor, then not visit it.
        for n in candidate_neighbors:
            if find_candidate(n, data) == -1:
                candidate_neighbors.remove(n)

    if len(candidate_neighbors) > 50:
        candidate_neighbors = sample(candidate_neighbors, 50)

    for n in candidate_neighbors: # candidate_neighbors without no neighbor nodes
        # calculate score
        h2 = data.embeddings[n]
        h2 = torch.Tensor(h2)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        action_value = cos(h1, h2)

        # action_value=torch.nn.functional.softmax((h1*(h2.t()))/(torch.norm(h1)*torch.norm(h2)))
        action_values.append(action_value)
        candidate_id.append(n)

    action_values = torch.Tensor(action_values)
    if len(action_values) >= args.k:
        topk = torch.topk(action_values, args.k)
        topk_value = topk[0]
        topk_index = topk[1]

        neighbour_id = []
        for n in topk_index:
            neighbour_id.append(candidate_id[n])
    else:
        neighbour_id = candidate_id
        topk_value = action_values
    # print('neighbour_id', neighbour_id)
    return neighbour_id, topk_value
    # return topk_index,topk_value


def action_to_index(action, topk_index, topk_value):
    action_1 = action.detach().numpy()
    action_2 = int(action_1)
    h = len(topk_index)
    if action_2 > h:
        action_3 = int(action_2 / ACTION_RANGE / 2 * h)
        next_state = topk_index[action_3-1]
        next_state_value = topk_value[action_3-1]
    else:
        next_state = topk_index[action_2-1]
        next_state_value = topk_value[action_2 - 1]
    # next_state=next_state[0]
    return next_state, next_state_value


def train_path(args, start, end, data):
    rl_planner = SMCP()
    # action_model=GaussianPolicy(100,args.k,256)
    # buffer=ReplayMemory(args.replay_buffer_size)

    refinefolder = args.refinefolder
    ic_relation_file = refinefolder + 'item_category.relation'
    ib_relation_file = refinefolder + 'item_brand.relation'
    ui_relation_file = refinefolder + 'user_item.relation'
    node_emb_dic = args.basedatafolder + 'nodewv.dic'
    user_history_file = args.userhistoryfile
    usize = args.usize
    isize = args.isize
    csize = args.csize
    bsize = args.bsize

    # because of using RL, don't need to pre-define meta-paths
    # ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    # ii_metapahts_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']

    outputinstancesfolder = args.outputinstancesfolder
    output_ui_paths_filename = outputinstancesfolder + 'ui.paths'
    output_ii_paths_filename = outputinstancesfolder + 'ii.paths'


    # The strategy to explore user-item paths and item-item paths is the same.

    userid = data.userid
    itemid = data.itemid
    categoryid = data.categoryid
    brandid = data.brandid

    # candidate={}

    # initial_state_emb=data.embeddings[start]
    # path=[]
    # path.append(start)

    candidate_neighbors = find_candidate(start, data)
    # if candidate_neighbors==-1:

    topk_index, topk_value = top_k(start, candidate_neighbors, data)

    # topk_id=[]
    # candidate_id[]
    # neighbors=[]
    # for m in topk_index:
    #     topk_id.append(candidate_id[m])

    # state=initial_state

    state = start
    state_emb = data.embeddings[state]
    done = 2

    reward = 1
    paths_list = []
    path_score_list = []
    no_neighobor = [] # do not visit these nodes again
    start_time = time.time()
    for epi in range(args.max_episode):
        epi_time = time.time()
        print(f'{epi} try time: {epi_time-start_time}')
        path = []
        values = []
        path.append(start)

        for k in range(args.max_length - 1):

            ##更新candidate_neighbours,topk_index

            # state 100,
            action, _ = rl_planner.policy.get_action(state_emb)

            # action=math.floor(action)
            # next_node=start+action
            # path.append(next_node)

            # next_state=data.embeddings[next_node]
            # state 直接是一个数
            next_state, next_state_value = action_to_index(action, topk_index, topk_value)

            next_state_emb = data.embeddings[next_state]

            path.append(next_state)
            values.append(next_state_value)
            # reward-=1

            if next_state == end:
                done = 1

                reward += 100

                # reward = torch.Tensor(reward)
                # done = torch.Tensor(done)
                reward = torch.from_numpy(np.array(reward))
                done = torch.from_numpy(np.array(done))
                next_state_emb = next_state_emb.squeeze()

                if type(state_emb) is np.ndarray:
                    state_emb = torch.Tensor(state_emb)
                state_emb = state_emb.detach().numpy()
                action = action.detach().numpy()
                reward = reward.detach().numpy()
                next_state_emb = next_state_emb.detach().numpy()
                done = done.detach().numpy()
                # print('push 1')
                # print(f'state_emb:{state_emb}')
                # print(f'action:{action}')
                # print(f'reward:{reward}')
                # print(f'next_state_emb:{next_state_emb}')
                # print(f'done:{done}')
                rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)
                print('end_path:', path)
                # path_score = np.prod(values)
                path_score = np.average(values)
                paths_list.append(path)
                path_score_list.append(path_score)
                break


            candidate_neighbors=find_candidate(next_state, data, no_neighobor)
            if k != 0 and k != args.max_length - 2:
                if candidate_neighbors != -1:
                    if end in candidate_neighbors:
                        done = 1
                        reward += 100
                        # reward = torch.Tensor(reward)
                        # done = torch.Tensor(done)
                        reward = torch.from_numpy(np.array(reward))
                        done = torch.from_numpy(np.array(done))
                        next_state_emb = next_state_emb.squeeze()

                        if type(state_emb) is np.ndarray:
                            state_emb = torch.Tensor(state_emb)
                        state_emb = state_emb.detach().numpy()
                        action = action.detach().numpy()
                        reward = reward.detach().numpy()
                        next_state_emb = next_state_emb.detach().numpy()
                        done = done.detach().numpy()

                        rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)
                        state = next_state
                        state_emb = data.embeddings[state]
                        end_emb = data.embeddings[end]
                        # score of state to end
                        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                        next_state_value = cos(torch.Tensor(state_emb), torch.Tensor(end_emb))

                        path.append(end)
                        values.append(next_state_value)
                        print('end_path:', path)
                        # path_score = np.prod(values)
                        path_score = np.average(values)
                        paths_list.append(path)
                        path_score_list.append(path_score)
                        break


            # reward = torch.Tensor(reward)
            reward = torch.from_numpy(np.array(reward))
            done = torch.from_numpy(np.array(done))
            next_state_emb = next_state_emb.squeeze()

            if type(state_emb) is np.ndarray:
                state_emb = torch.Tensor(state_emb)
            state_emb = state_emb.detach().numpy()
            action = action.detach().numpy()
            reward = reward.detach().numpy()
            next_state_emb = next_state_emb.detach().numpy()
            done = done.detach().numpy()

            state_1 = state_emb
            # print('push 3')
            # print(f'state_emb:{state_emb}')
            # print(f'action:{action}')
            # print(f'reward:{reward}')
            # print(f'next_state_emb:{next_state_emb}')
            # print(f'done:{done}')
            rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)

            state = next_state



            state_emb = data.embeddings[state]


            # candidate_neighbors = find_candidate(state, data, no_neighobor)
            # if candidate_neighbors != -1:
            #     # delete nodes with no neighbors
            #     if len(candidate_neighbors) < 50: # because candidate_neighbors sometime is an int
            #         for n in candidate_neighbors:
            #             if find_candidate(n, data) == -1:
            #                 candidate_neighbors.remove(n)

            if candidate_neighbors == -1 or len(candidate_neighbors) == 0:
                no_neighobor.append(state)
                reward -= 10
                done = 1
                # print('push 4')
                # print(f'state_emb:{state_emb}')
                # print(f'action:{action}')
                # print(f'reward:{reward}')
                # print(f'next_state_emb:{next_state_emb}')
                # print(f'done:{done}')
                rl_planner.buffer.push(state_1, action, reward, state_emb, done)
                break

            topk_index, topk_value = top_k(state, candidate_neighbors, data)

            if k == args.max_length - 1:
                if state != end:
                    reward -= 100

            if len(rl_planner.buffer) > args.batch_size:
                rl_planner.soft_q_update()

        print('path:', path)

    return paths_list, path_score_list


def get_path(args, start, end, data):
    paths_list, path_score_list = train_path(args, start, end, data)
    rl_planner = SMCP()

    state = start
    state_emb = data.embeddings[state]
    done = 2

    reward = 1

    candidate_neighbors = find_candidate(start, data)

    topk_index, topk_value = top_k(start, candidate_neighbors, data)

    good_path = []
    values = []

    for k in range(args.max_length - 1):
        # state 100,

        path = []
        scores = []
        path.append(start)

        ##更新candidate_neighbours,topk_index

        # state 100,
        action, _ = rl_planner.policy.get_action(state_emb)

        # action=math.floor(action)
        # next_node=start+action
        # path.append(next_node)

        # next_state=data.embeddings[next_node]
        # state 直接是一个数
        next_state, next_state_value = action_to_index(action, topk_index, topk_value)
        # next_state_1=next_state[0]
        # next_state_emb=data.embeddings[next_state_1]
        next_state_emb = data.embeddings[next_state]
        path.append(next_state)
        values.append(next_state_value)
        # reward-=1

        if next_state == end:
            done = 1

            reward += 10

            reward = torch.from_numpy(np.array(reward))
            # done = torch.Tensor(done)
            done = torch.from_numpy(np.array(done))
            next_state_emb = next_state_emb.squeeze()

            state_emb = state_emb.detach().numpy()
            action = action.detach().numpy()
            reward = reward.detach().numpy()
            next_state_emb = next_state_emb.detach().numpy()
            done = done.detach().numpy()
            # print('push 5')
            # print(f'state_emb:{state_emb}')
            # print(f'action:{action}')
            # print(f'reward:{reward}')
            # print(f'next_state_emb:{next_state_emb}')
            # print(f'done:{done}')
            rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)
            print('end_path:', path)
            good_path.append(path)
            # path_score = np.prod(values)
            path_score = np.average(values)
            paths_list.append(path)
            path_score_list.append(path_score)
            break

        candidate_neighbors = find_candidate(next_state, data)
        if k != 0 and k != args.max_length - 2:
            if candidate_neighbors != -1:
                if end in candidate_neighbors:
                    done = 1
                    reward += 100
                    # reward = torch.Tensor(reward)
                    # done = torch.Tensor(done)
                    reward = torch.from_numpy(np.array(reward))
                    done = torch.from_numpy(np.array(done))
                    next_state_emb = next_state_emb.squeeze()

                    if type(state_emb) is np.ndarray:
                        state_emb = torch.Tensor(state_emb)
                    state_emb = state_emb.detach().numpy()
                    action = action.detach().numpy()
                    reward = reward.detach().numpy()
                    next_state_emb = next_state_emb.detach().numpy()
                    done = done.detach().numpy()
                    rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)
                    state = next_state
                    state_emb = data.embeddings[state]
                    end_emb = data.embeddings[end]
                    # score of state to end
                    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                    next_state_value = cos(torch.Tensor(state_emb), torch.Tensor(end_emb))

                    path.append(end)
                    values.append(next_state_value)
                    print('end_path:', path)
                    # path_score = np.prod(values)
                    path_score = np.average(values)
                    paths_list.append(path)
                    path_score_list.append(path_score)
                    break

        reward = torch.from_numpy(np.array(reward))
        done = torch.from_numpy(np.array(done))
        next_state_emb = next_state_emb.squeeze()

        state_emb = state_emb.detach().numpy()
        action = action.detach().numpy()
        reward = reward.detach().numpy()
        next_state_emb = next_state_emb.detach().numpy()
        done = done.detach().numpy()

        rl_planner.buffer.push(state_emb, action, reward, next_state_emb, done)

        state = next_state

        state_emb = data.embeddings[state]

        # candidate_neighbors = find_candidate(state, data)
        # delete nodes with no neighbors
        # if candidate_neighbors != -1:
        #     if len(candidate_neighbors) < 50:
        #         for n in candidate_neighbors:
        #             if find_candidate(n, data) == -1:
        #                 candidate_neighbors.remove(n)

        if candidate_neighbors == -1 or len(candidate_neighbors) == 0:
            # no_neighobor.append(state)
            break

        topk_index, topk_value = top_k(state, candidate_neighbors, data)

        if k == args.max_length - 1:
            if state != end:
                reward -= 100

        if len(rl_planner.buffer) > args.batch_size:
            rl_planner.soft_q_update()

    print('get path:', path)

    # return good_path
    return paths_list, path_score_list


def ui_path(args):
    refinefolder = args.refinefolder
    ic_relation_file = refinefolder + 'item_category.relation'
    ib_relation_file = refinefolder + 'item_brand.relation'
    ui_relation_file = refinefolder + 'user_item.relation'
    node_emb_dic = args.basedatafolder + 'nodewv.dic'
    usize = args.usize
    isize = args.isize
    csize = args.csize
    bsize = args.bsize

    outputinstancesfolder = args.outputinstancesfolder
    output_ui_paths_filename = outputinstancesfolder + 'ui.paths'+'.'+str(args.start_user)+'-'+str(args.end_user)


    # load data
    data = Prepared_data(ib_relation_file=ib_relation_file, ic_relation_file=ic_relation_file,
                         ui_relation_file=ui_relation_file, node_emb_dic=node_emb_dic,
                         usize=usize, isize=isize, csize=csize, bsize=bsize)

    userid = data.userid
    itemid = data.itemid
    categoryid = data.categoryid
    brandid = data.brandid
    ui_file = open(output_ui_paths_filename, 'w+')

    part_user_id = torch.arange(args.start_user, args.end_user)
    for m in part_user_id:
        start = m.item()

        if find_candidate(start, data) == -1:
            break

        for n in data.ui_dict[start]:
            end = n
            print(f'\n\nstart:{start}\t end:{end}')


            ## [[1,2],[4,5,6,7],[7,5,3]]  good path
            
            # good_path = get_path(args, start, end, data)
            paths_list, path_score_list = get_path(args, start, end, data)
            paths_list_unique = []
            path_score_list_unique = []
            #delete same paths
            for index,i in enumerate(paths_list):
                if i not in paths_list_unique:
                    paths_list_unique.append(i)
                    path_score_list_unique.append(path_score_list[index])
            # zip(paths_list, path_score_list) and order
            zipped_paths_scores = list(zip(paths_list_unique, path_score_list_unique))
            res = sorted(zipped_paths_scores, key=lambda x: x[1],reverse=True)

            if len(paths_list_unique) >= args.top_n_paths:
                res = res[:args.top_n_paths]
            print(res)

            res_len = len(res)
            if res_len > 0:
                ui_file.write(str(start) + ',' + str(end) + '\t')
                ui_file.write(str(res_len)+'\t')
                for path,score in res:
                    path_str = [str(i) for i in path]
                    ui_file.write(' '.join(path_str))
                    ui_file.write('\t')
                ui_file.write('\n')
    ui_file.close()

def ii_path(args):
    refinefolder = args.refinefolder
    ic_relation_file = refinefolder + 'item_category.relation'
    ib_relation_file = refinefolder + 'item_brand.relation'
    ui_relation_file = refinefolder + 'user_item.relation'
    node_emb_dic = args.basedatafolder + 'nodewv.dic'
    usize = args.usize
    isize = args.isize
    csize = args.csize
    bsize = args.bsize


    outputinstancesfolder = args.outputinstancesfolder
    output_ii_paths_filename = outputinstancesfolder + 'ii.paths' + '.' + str(args.start_user) + '-' + str(
        args.end_user)

    # load data
    data = Prepared_data(ib_relation_file=ib_relation_file, ic_relation_file=ic_relation_file,
                         ui_relation_file=ui_relation_file, node_emb_dic=node_emb_dic,
                         usize=usize, isize=isize, csize=csize, bsize=bsize)

    userid = data.userid
    itemid = data.itemid
    categoryid = data.categoryid
    brandid = data.brandid
    ii_file = open(output_ii_paths_filename, 'w+')

    part_user_id = torch.arange(args.start_user, args.end_user)
    for user in part_user_id:
        user = user.item()

        num_item = len(data.ui_dict[user])
        item = data.ui_dict[user]
        for m in range(num_item - 1):
            start = item[m]
            end = item[m + 1]
            print(f'\n\nuser: {user}\t start item: {start}\t end item: {end}')

            if find_candidate(start, data) == -1:
                break
            ## [[1,2],[4,5,6,7],[7,5,3]]  good path

            paths_list, path_score_list = get_path(args, start, end, data)
            paths_list_unique = []
            path_score_list_unique = []
            # delete same paths
            for index, i in enumerate(paths_list):
                if i not in paths_list_unique:
                    paths_list_unique.append(i)
                    path_score_list_unique.append(path_score_list[index])
            # zip(paths_list, path_score_list) and order
            zipped_paths_scores = list(zip(paths_list_unique, path_score_list_unique))
            res = sorted(zipped_paths_scores, key=lambda x: x[1], reverse=True)

            if len(paths_list_unique) >= args.top_n_paths:
                res = res[:args.top_n_paths]
            print(res)

            res_len = len(res)
            if res_len > 0:
                ii_file.write(str(start) + ',' + str(end) + '\t')
                ii_file.write(str(res_len) + '\t')
                for path, score in res:
                    path_str = [str(i) for i in path]
                    ii_file.write(' '.join(path_str))
                    ii_file.write('\t')
                ii_file.write('\n')
    ii_file.close()


    # # for each user, explore user-item paths using RL
    # for userid in data.userid:
    #     state=data.embeddings[userid]
    #     neigh_range=[]

    #     neigh_range.append(data.load_ui)

    #     for itemid in data.ui_dict[userid]:

    #         # for each user, for each user's item, we explore paths between these two specific nodes
    #         # do RL, start node: userid , target node: itemid

    #         # todo: this part should get learned paths from userid to itemid. Each path should have a score. (see D. in III. in draft paper)
    #         # the paths should be saved in `output_ui_paths_filename`, the format is like `data/Goodreads_mystery/path/all*/uibi.paths`
    #         # descending score and get top q=5 path from total p paths (if p<=5, then get p paths)
    #         raise NotImplementedError
    #         # this part should be
    #         # RL_explore = RL_class()
    #         # RL_explore.explore_path(start_node = userid, end_node = itemid, ...)

    # # for each user, explore item-item paths using RL
    # for userid in range(usize):
    #     num_item = len(data.ui_dict[userid])
    #     for item_index in range(num_item - 1):
    #         item_1 = data.ui_dict[userid][item_index]
    #         item_2 = data.ui_dict[userid][item_index + 1]
    #         # for each user, for each user's item, we explore paths between these two specific nodes
    #         # do RL, start node: item_1 , target node: item_2

    #         # todo: this part should get learned paths from item_1 to item_2. Each path should have a score. (see D. in III. in draft paper)
    #         # the paths should be saved in `output_ii_paths_filename`, the format is like `data/Goodreads_mystery/path/all*/uibi.paths`
    #         # descending score and get top q=5 path from total p paths (if p<=5, then get p paths)
    #         raise NotImplementedError
    #         # this part should be
    #         # RL_explore = RL_class()
    #         # RL_explore.explore_path(start_node = item_1, end_node = item_2, ...)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate user item and item item instances')
    parser.add_argument('start_user', type=int, default=0, nargs='?')
    parser.add_argument('end_user', type=int, default=11800, nargs='?')
    parser.add_argument('path_type', type=str, default='user',
                        nargs='?', help='user for user-item paths; item for item-item paths')

    parser.add_argument('basedatafolder', type=str, default='../Goodreads_mystery/',
                        nargs='?', help='this data base folder')
    parser.add_argument('outputinstancesfolder', type=str,
                        default='../Goodreads_mystery/path/all_ui_ii_instance_paths/',
                        nargs='?', help='this instances folder')
    parser.add_argument('refinefolder', type=str, default='../Goodreads_mystery/refine/',
                        nargs='?',
                        help='output to refine folder')
    parser.add_argument('userhistoryfile', type=str,
                        default='../Goodreads_mystery/path/user_history/user_history.txt',
                        nargs='?',
                        help='user history file')

    # parser.add_argument('basedatafolder', type=str, default='../Goodreads_mystery/',
    #                     nargs='?', help='this data base folder')
    # parser.add_argument('outputinstancesfolder', type=str,
    #                     default='../Goodreads_mystery/path/all_ui_ii_instance_paths/',
    #                     nargs='?', help='this instances folder')
    # parser.add_argument('refinefolder', type=str, default='../Goodreads_mystery/refine/',
    #                     nargs='?',
    #                     help='output to refine folder')
    # parser.add_argument('userhistoryfile', type=str,
    #                     default='../Goodreads_mystery/path/user_history/user_history.txt',
    #                     nargs='?',
    #                     help='user history file')
    parser.add_argument('usize', type=int, default=11800, nargs='?')
    parser.add_argument('isize', type=int, default=12142, nargs='?')
    parser.add_argument('csize', type=int, default=3633, nargs='?')
    parser.add_argument('bsize', type=int, default=1442, nargs='?')

    parser.add_argument('k', type=int, default=10, nargs='?')
    parser.add_argument('top_n_paths', type=int, default=10, nargs='?')
    parser.add_argument('max_length', type=int, default=6, nargs='?')
    parser.add_argument('replay_buffer_size', type=int, default=100000, nargs='?')
    parser.add_argument('max_episode', type=int, default=50, nargs='?')

    parser.add_argument('batch_size', type=int, default=1024, nargs='?')
    args = parser.parse_args()

    Path(args.outputinstancesfolder).mkdir(parents=True, exist_ok=True)

    # train_path(args,start=32,end=17800)
    # final=get_path(args,start=32,end=17800)
    # print('final',final)
    if args.path_type == 'user':
        ui_path(args)
    elif args.path_type == 'item':
        ii_path(args)
    else:
        print('Please enter correct path type: user or item')


    # Amazon_Music
    # parser.add_argument('usize', type=int, default=1450, nargs='?')
    # parser.add_argument('isize', type=int, default=11457, nargs='?')
    # parser.add_argument('csize', type=int, default=429, nargs='?')
    # parser.add_argument('bsize', type=int, default=1185, nargs='?')


    # # Goodreads_mystery
    # # parser.add_argument('usize', type=int, default=4600, nargs='?')
    # # parser.add_argument('isize', type=int, default=36663, nargs='?')
    # # parser.add_argument('csize', type=int, default=1592, nargs='?')
    # # parser.add_argument('bsize', type=int, default=3790, nargs='?')
    # # Goodreads_mystery
    # parser.add_argument('usize', type=int, default=9300, nargs='?')
    # parser.add_argument('isize', type=int, default=58743, nargs='?')
    # parser.add_argument('csize', type=int, default=820, nargs='?')
    # parser.add_argument('bsize', type=int, default=5404, nargs='?')

    # # Goodreads
    # parser.add_argument('usize', type=int, default=11800, nargs='?')
    # parser.add_argument('isize', type=int, default=12142, nargs='?')
    # parser.add_argument('asize', type=int, default=3633, nargs='?')
    # parser.add_argument('psize', type=int, default=1442, nargs='?')
