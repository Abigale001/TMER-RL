#encoding=utf-8
import time
import math
from torchnlp.nn import Attention
import torch.utils.data as Data
from data.path.path_attention.att import *
from data.data_utils import *
import pickle
import os
import io
import eva
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'run.py device: {device}')


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class Recommendation(nn.Module):
    def __init__(self, in_features):
        """

        :param in_features: mlp input latent: here 100
        :param out_features:  mlp classification number, here neg+1
        """
        super(Recommendation, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(2, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(2))
        self.in_features = in_features
        self.attention1 = Attention(self.in_features)
        self.attention2 = Attention(self.in_features)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, item_emb, sequence_emb, ifTrain):
        """

        :param sequence_emb
        :return:
        """
        x, weights = self.attention1(item_emb, sequence_emb)
        output = F.linear(x, self.weight, self.bias)
        a, b, c = output.shape
        output = output.reshape((a, c))
        if ifTrain:
            fe = F.log_softmax(output)
            return fe
        else:
            fe = F.softmax(output)
            # fe = F.normalize(s)
            return fe

def instances_slf_att(input_tensor):
    instances_slf_att = Self_Attention_Network(user_item_dim=latent_size).to(device)
    distance_slf_att = nn.MSELoss()
    optimizer_slf_att = torch.optim.Adam(instances_slf_att.parameters(), lr=0.01, weight_decay=0.00005)
    num_epochs_slf_att = 50
    for epoch in range(num_epochs_slf_att):
        output = instances_slf_att(input_tensor.to(device))
        loss_slf = distance_slf_att(output.to(device), input_tensor.to(device)).to(device)
        optimizer_slf_att.zero_grad()
        loss_slf.backward()
        optimizer_slf_att.step()
    slf_att_embeddings = output.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return slf_att_embeddings

def item_attention(item_input, ii_path):
    """
    here are two item attention.
    the first item attention: ii_path, last_item_att_output
    the second item attention: ii_path, this_item_input
    :param ii_path:
    :param last_item_att_output:
    :param this_item_input:
    :return: item att output
    """
    item_atten = ItemAttention(latent_dim=ii_path.shape[-1], att_size=100).to(device)
    distance_att = nn.MSELoss()
    optimizer_att = torch.optim.Adam(item_atten.parameters(), lr=0.01, weight_decay=0.00005)
    num_epoch = 10
    for epoch in range(num_epoch):
        output = item_atten(item_input.to(device), ii_path.to(device)).to(device)
        loss_slf = distance_att(output, item_input.to(device))
        optimizer_att.zero_grad()
        loss_slf.backward()
        optimizer_att.step()

    att_embeddings = output.detach().cpu().numpy() # [1,100]
    torch.cuda.empty_cache()
    return att_embeddings

def rec_net(train_loader, test_loader, node_emb, sequence_tensor):
    best_hit_1 = 0.0
    best_hit_5 = 0.0
    best_hit_10 = 0.0
    best_hit_20 = 0.0
    best_ndcg_1 = 0.0
    best_ndcg_5 = 0.0
    best_ndcg_10 = 0.0
    best_ndcg_20 = 0.0
    recommendation = Recommendation(100).to(device)

    optimizer = torch.optim.Adam(recommendation.parameters(), lr=1e-4)
    for epoch in range(100):
        train_start_time = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch_item_emb = node_emb[batch[:, 1]].reshape((batch.shape[0], 1, 100)).to(device)
            batch_labels = batch[:, 2].to(device)
            batch_sequence_tensor = sequence_tensor[batch[:,0]].reshape((batch.shape[0], 9, 100)).to(device)
            optimizer.zero_grad()
            prediction = recommendation(batch_item_emb, batch_sequence_tensor, True).to(device)
            loss_train = torch.nn.functional.cross_entropy(prediction, batch_labels).to(device)
            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()
        train_time = time.time() - train_start_time
        print(f'epoch: {epoch}, training loss: {running_loss}, train time: {train_time}')


        testing_start_time = time.time()

        all_hits = {1: 0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
        all_mrr = {1: 0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
        all_ndcg = {1: 0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
        all_pos_num = 0.0
        for step, batch in enumerate(test_loader):
            batch_item_emb = node_emb[batch[:, 1]].reshape((batch.shape[0], 1, 100)).to(device)
            batch_sequence_tensor = sequence_tensor[batch[:,0]].reshape((batch.shape[0], 9, 100)).to(device)
            prediction = recommendation(batch_item_emb, batch_sequence_tensor, True)[:, -1].to(device) # 50100 # 1,0,0,0,,,,,1,0,0,0,,,,,
            prediction = prediction.detach().cpu()
            pos_index_num = int(prediction.shape[0]/(1 + negative_num))

            # calculate hr and ndcg
            for pos_index in range(pos_index_num): # test_batch
                all_pos_num += 1.0
                # for each 501 pairs (1 pos pair + 500 neg pairs)
                this_pair_scores = prediction[pos_index*(negative_num+1):(pos_index+1)*(negative_num+1)].tolist()

                # this_pair_score is the score for the neg 500 + 1 pos
                s = np.array(this_pair_scores)
                sorted_s = np.argsort(-s) # array[500,2,5,394,...]
                hit_s = (sorted_s >= negative_num).astype(int)

                ground_truth_id = [np.argwhere(hit_s == 1).item()]
                for k in [1,5,10,20]:
                    all_hits[k] += eva.hit_at_k(hit_s, k)
                    all_ndcg[k] += eva.ndcg_at_k(hit_s, k, ground_truth_id, 0)


        hit1_avg = all_hits[1] / float(all_pos_num)
        hit5_avg = all_hits[5] / float(all_pos_num)
        hit10_avg = all_hits[10] / float(all_pos_num)
        hit20_avg = all_hits[20] / float(all_pos_num)

        ndcg1_avg = all_ndcg[1] / float(all_pos_num)
        ndcg5_avg = all_ndcg[5] / float(all_pos_num)
        ndcg10_avg = all_ndcg[10] / float(all_pos_num)
        ndcg20_avg = all_ndcg[20] / float(all_pos_num)
        testing_time = time.time() - testing_start_time

        if best_hit_1 < hit1_avg:
            best_hit_1 = hit1_avg
        if best_hit_5 < hit5_avg:
            best_hit_5 = hit5_avg
        if best_hit_10 < hit10_avg:
            best_hit_10 = hit10_avg
        if best_hit_20 < hit20_avg:
            best_hit_20 = hit20_avg
        if best_ndcg_1 < ndcg1_avg:
            best_ndcg_1 = ndcg1_avg
        if best_ndcg_5 < ndcg5_avg:
            best_ndcg_5 = ndcg5_avg
        if best_ndcg_10 < ndcg10_avg:
            best_ndcg_10 = ndcg10_avg
        if best_ndcg_20 < ndcg20_avg:
            best_ndcg_20 = ndcg20_avg

        print(f"epo:{epoch}|"
              f"HR@1:{hit1_avg:.4f} | HR@5:{hit5_avg:.4f} | HR@10:{hit10_avg:.4f} | HR@20:{hit20_avg:.4f} |"
              f" NDCG@1:{ndcg1_avg:.4f} | NDCG@5:{ndcg5_avg:.4f} | NDCG@10:{ndcg10_avg:.4f}| NDCG@20:{ndcg20_avg:.4f} |"
              f" best_HR@1:{best_hit_1:.4f} | best_HR@5:{best_hit_5:.4f} | best_HR@10:{best_hit_10:.4f} | best_HR@20:{best_hit_20:.4f} |"
              f" best_NDCG@1:{best_ndcg_1:.4f} | best_NDCG@5:{best_ndcg_5:.4f} | best_NDCG@10:{best_ndcg_10:.4f} | best_NDCG@20:{best_ndcg_20:.4f} |"
              f" train_time:{train_time:.2f} | test_time:{testing_time:.2f}")



    print('training finish')

if __name__ == '__main__':

    att_size = 100
    latent_size = 100
    negative_num = 500
    user_n_items = 4 # for each user, it has n items

    data_name = 'Amazon_Music'
    # split train and test data
    user_history_file = 'data/'+data_name+'/path/user_history/user_history.txt'

    # get train and test links
    N = negative_num  # for each user item, there are N negative samples.
    train_file = 'data/'+data_name+'/links/training_neg_' + str(N) + '.links'
    test_file= 'data/'+data_name+'/links/testing_neg_' + str(N) + '.new.links'
    train_data, test_data = load_train_test_data(train_file, test_file)

    # load users id and items id
    maptype2id_file = 'data/'+data_name+'/refine/map.type2id'
    type2id = pickle.load(open(maptype2id_file, 'rb'))
    users_list = type2id['user']
    user_num = len(users_list)
    items_list = type2id['item']
    item_num = len(items_list)

    # load node embeds
    node_emb_file = 'data/'+data_name+'/nodewv.dic'
    node_emb = load_node_tensor(node_emb_file)

    # load ui pairs
    ui_dict = load_ui_seq_relation(user_history_file)

    # load all ui embeddings and ii embeddings
    metapath_emb_folder = 'data/'+data_name+'/path/meta_path_instances_representation/'
    user_item_direct_emb_file = 'data/'+data_name+'/representations/user_item_dic.wv'
    user_item_direct_emb = CPU_Unpickler(open(user_item_direct_emb_file, 'rb')).load()
    item_item_direct_emb_file = 'data/'+data_name+'/path/user_history/item_item.wv'
    item_item_direct_emb = load_item_item_wv(item_item_direct_emb_file)
    ui_all_paths_emb = load_ui_instances_emb(metapath_emb_folder, user_num, ui_dict, user_item_direct_emb)
    edges_id_dict_file = 'data/'+data_name+'/path/user_history/user_history.edges2id'
    edges_id_dict = pickle.load(open(edges_id_dict_file, 'rb'))
    ii_all_paths_emb = load_ii_metapath_instances_emb(metapath_emb_folder, user_num, ui_dict, item_item_direct_emb, edges_id_dict)
    labels = train_data[:, 2].to(device)
    print(f'labels.shape: {labels.shape}')
    print('loading node embedding, all user-item and item-item paths embedding...finished')



    # 1. user-item instances slf attention and for each user item, get one instance embedding.
    print('start training user-item instance self attention module...')
    maxpool = Maxpooling()
    ui_paths_att_emb = defaultdict()
    t = time.time()
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - t
            print('user ',u, 'time: ',t_here)
        user_item_paths_emb = ui_all_paths_emb[u]
        this_user_ui_paths_att_emb = defaultdict()
        item_index = 0
        # for i in ui_dict[u][:4]: # because there are 4 items for each user in the training data
        for i in ui_dict[u]:
            item_index += 1
            if len(ui_all_paths_emb[u][(u, i)]) == 1:
                this_user_ui_paths_att_emb[(u, i)] = ui_all_paths_emb[u][(u, i)]
            else:
                slf_att_input = torch.Tensor(ui_all_paths_emb[u][(u, i)]).unsqueeze(0)
                this_user_ui_paths_att_emb[(u, i)] = instances_slf_att(slf_att_input)
                # user-item instances to one. for each user-item pair, only one instance is needed.
                max_pooling_input = torch.from_numpy(this_user_ui_paths_att_emb[(u, i)])
                get_one_ui = maxpool(max_pooling_input).squeeze(0)
                this_user_ui_paths_att_emb[(u, i)] = get_one_ui
        ui_paths_att_emb[u] = this_user_ui_paths_att_emb
    ui_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) +'_ui_batch_paths_att_emb.pkl'
    pickle.dump(ui_paths_att_emb, open(ui_batch_paths_att_emb_pkl_file, 'wb'))

    # 2. item-item instances slf attention
    print('start training item-item instance self attention module...')
    start_t_ii = time.time()
    ii_paths_att_emb = defaultdict()
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - start_t_ii
            print('user ',u, 'time: ',t_here)
        item_item_paths_emb = ii_all_paths_emb[u]
        num_item = len(ui_dict[u])
        this_user_ii_paths_att_emb = defaultdict()
        item_index = 0
        for i_index in range(num_item - 1):
        # for i_index in range(4 - 1):
            item_index += 1
            i1 = ui_dict[u][i_index]
            i2 = ui_dict[u][i_index + 1]
            if len(ii_all_paths_emb[u][(i1, i2)]) == 1:
                this_user_ii_paths_att_emb[(i1, i2)] = ii_all_paths_emb[u][(i1, i2)]
            else:
                slf_att_input = torch.Tensor(ii_all_paths_emb[u][(i1, i2)]).unsqueeze(0)
                this_user_ii_paths_att_emb[(i1, i2)] = instances_slf_att(slf_att_input).squeeze(0)
                this_user_ii_paths_att_emb[(i1, i2)] = torch.from_numpy(this_user_ii_paths_att_emb[(i1, i2)])
        ii_paths_att_emb[u] = this_user_ii_paths_att_emb
    ii_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    pickle.dump(ii_paths_att_emb, open(ii_batch_paths_att_emb_pkl_file, 'wb'))



    # 3. user and item embedding
    ii_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    ui_batch_paths_att_emb_pkl_file = data_name + '_' + str(negative_num) + '_ui_batch_paths_att_emb.pkl'
    # ii_paths_att_emb = pickle.load(open(ii_batch_paths_att_emb_pkl_file, 'rb'))
    ii_paths_att_emb = CPU_Unpickler(open(ii_batch_paths_att_emb_pkl_file, 'rb')).load()
    # ui_paths_att_emb = torch.load(io.BytesIO(open(ui_batch_paths_att_emb_pkl_file, 'rb')), map_location=torch.device('cpu'))
    ui_paths_att_emb = CPU_Unpickler(open(ui_batch_paths_att_emb_pkl_file, 'rb')).load()
    # ui_paths_att_emb = torch.load(ui_batch_paths_att_emb_pkl_file, map_location='cpu', pickle_module=pickle)
    # ui_paths_att_emb = pickle.load(open(ui_batch_paths_att_emb_pkl_file, 'rb'))
    print('start updating user and item embedding...')
    start_t_u_i = time.time()
    sequence_concat = []
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - start_t_u_i
            print('user ',u, 'time: ',t_here)
        user_sequence_concat = defaultdict()
        this_user_ui_paths_dic = ui_paths_att_emb[u]
        this_user_ii_paths_dic = ii_paths_att_emb[u]
        # for user uid, item1
        u_emb = node_emb[u].reshape((1, -1)).to(device)
        i1_id = ui_dict[u][0]
        u_i1_emb = this_user_ui_paths_dic[(u, i1_id)].reshape((1, -1)).to(device)
        item1_emb = node_emb[i1_id].reshape((1, -1))
        # input: u_i1_emb, item1_emb   after attention: the same dimension
        item1_att = item_attention(item1_emb, u_i1_emb.unsqueeze(0)).reshape((1, -1))
        item1_att = torch.from_numpy(item1_att).to(device)
    
    
        user_sequence_concat[0] = torch.cat([u_emb, u_i1_emb, item1_att], dim=0).to(device)
    
        last_item_att = item1_att
        for i_index in range(1, user_n_items):
            i1 = ui_dict[u][i_index - 1]
            i2 = ui_dict[u][i_index]
            item_att_input = this_user_ii_paths_dic[(i1, i2)].unsqueeze(0)
            ii_1 = item_attention(last_item_att, item_att_input).reshape((1, -1))
            ii_1 = torch.from_numpy(ii_1).to(device)
            ii_2 = item_attention(node_emb[i2].unsqueeze(0), item_att_input).reshape((1, -1))
            ii_2 = torch.from_numpy(ii_2).to(device)
            user_sequence_concat[i_index] = torch.cat([u_emb, ii_1, ii_2], dim=0)
            last_item_att = ii_2
        sequence_concat.append(torch.cat([user_sequence_concat[i] for i in range(0, user_n_items - 1)], 0))
    sequence_tensor = torch.stack(sequence_concat)
    sequence_tensor_pkl_name = data_name + '_' + str(negative_num) + '_sequence_tensor.pkl'
    pickle.dump(sequence_tensor, open(sequence_tensor_pkl_name, 'wb'))

    # 4. recommendation
    print('start training recommendation module...')
    sequence_tensor_pkl_name = data_name + '_' + str(negative_num) + '_sequence_tensor.pkl'
    sequence_tensor = pickle.load(open(sequence_tensor_pkl_name, 'rb'))
    item_emb = node_emb[user_num:(user_num+item_num),:]
    BATCH_SIZE = 100
    test_BATCH_SIZE = 500*(negative_num+1)

    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  #
        num_workers=10,  #
    )
    test_loader = Data.DataLoader(
        dataset=test_data,  # torch TensorDataset format
        batch_size=test_BATCH_SIZE,  # mini batch size
        shuffle=False,  #
        num_workers=10,  #
    )
    rec_net(train_loader, test_loader, node_emb, sequence_tensor)
