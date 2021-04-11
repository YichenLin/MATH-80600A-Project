import os
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

import numpy as np
import torch as th
import dgl 
import pandas as pd

class MetricLogger(object):
    def __init__(self, save_dir, log_interval):
        self.save_dir = save_dir
        self.log_interval = log_interval

    def log(self, info, model, optimizer):
        epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
                epoch, train_loss, test_rmse))
        # if type(epoch) == int and epoch % self.log_interval == 0:
        #     print('Saving model states...')
        #     model_name = os.path.join(self.save_dir, 'model_checkpoint{}.pth'.format(epoch))
        #     optimizer_name = os.path.join(
        #         self.save_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        #     )
        #     if model is not None:
        #         th.save(model.state_dict(), model_name)
        #     if optimizer is not None:
        #         th.save(optimizer.state_dict(), optimizer_name)

        
def load_official_trainvaltest_split(dataset, testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """

    sep = '\t'

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'data/' + fname


    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    filename_train = 'data/' + dataset + '/u1.base'
    filename_test = 'data/' + dataset + '/u1.test'

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio*len(data_array_train))]]

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train+num_val]
    idx_nonzero_test = idx_nonzero[num_train+num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
    pairs_nonzero_test = pairs_nonzero[num_train+num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(1234)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])
    
    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if dataset =='ml-100k':

        # movie features (genres)
        sep = r'|'
        movie_file = 'data/' + dataset + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # user features

        sep = r'|'
        users_file = 'data/' + dataset + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        age = users_df['age'].values
        age_max = age.max()

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: "+str(u_features.shape))
    print("Item features shape: "+str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values        
        
def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])

def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str

def MinMaxScaling(x, axis=0):
    dist = x.max(axis=axis) - x.min(axis=axis)
    x = (x - x.min(axis=axis)) / (dist + 1e-7)
    return x

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x

def cal_dist(csr_graph, node_to_remove):
    # cal dist to node 0, with target edge nodes 0/1 removed
    nodes = list(set(range(csr_graph.shape[1])) - set([node_to_remove]))
    csr_graph = csr_graph[nodes, :][:, nodes]
    dists = np.clip(sp.csgraph.dijkstra(
                        csr_graph, indices=0, directed=False, unweighted=True, limit=1e6
                    )[1:], 0, 1e7)
    return dists.astype(np.int64)

def get_neighbor_nodes_labels(ind, graph, mode="bipartite",
        hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    
    if mode=="bipartite":
        # 1. neighbor nodes sampling
        dist = 0
        u_nodes, v_nodes = ind[0].unsqueeze(dim=0), ind[1].unsqueeze(dim=0)
        u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
        u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

        for dist in range(1, hop+1):
            # sample neigh alternately
            u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.in_edges(u_fringe)[0]
            u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = th.unique(th.cat([u_visited, u_fringe]))
            v_visited = th.unique(th.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = th.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
 
        nodes = th.cat([u_nodes, v_nodes])

        # 2. node labeling
        u_node_labels = th.stack([x*2 for x in u_dist])
        v_node_labels = th.stack([x*2+1 for x in v_dist])
        node_labels = th.cat([u_node_labels, v_node_labels])
    
    elif mode=="homo": 
        # 1. neighbor nodes sampling
        dist = 0
        nodes = th.stack(ind)
        dists = th.zeros_like(nodes) 
        visited = th.unique(nodes)
        fringe = th.unique(nodes)

        for dist in range(1, hop+1):
            fringe = graph.in_edges(fringe)[0]    
            fringe = th.from_numpy(np.setdiff1d(fringe.numpy(), visited.numpy()))
            visited = th.unique(th.cat([visited, fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:int(sample_ratio*len(fringe))]]
            if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(fringe) == 0:
                break
            nodes = th.cat([nodes, fringe])
            dists = th.cat([dists, th.full((len(fringe), ), dist, dtype=th.int64)])
        
        # 2. node labeling
        node_labels = dists
    
    elif mode=="grail":
        # 1. neighbor nodes sampling
        # make sure ind not in uv nodes.
        u_nodes, v_nodes = th.tensor([], dtype=th.long), th.tensor([], dtype=th.long)
        # u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.tensor([ind[0]]), th.tensor([ind[1]])
        u_fringe, v_fringe = th.tensor([ind[0]]), th.tensor([ind[1]])

        for dist in range(1, hop+1):
            # sample neigh separately
            u_fringe = graph.in_edges(u_fringe)[0]
            v_fringe = graph.in_edges(v_fringe)[0]

            u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = th.unique(th.cat([u_visited, u_fringe]))
            v_visited = th.unique(th.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = th.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            # u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            # v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
    
        nodes = th.from_numpy(np.intersect1d(u_nodes.numpy(), v_nodes.numpy()))
        # concatenate ind to front, and node labels of ind can be added easily.
        nodes = th.cat([ind, nodes])
       
        # 2. node labeling
        csr_subgraph = graph.subgraph(nodes).adjacency_matrix_scipy(return_edge_ids=False)
        dists = th.stack([th.tensor(cal_dist(csr_subgraph, 1)), 
                          th.tensor(cal_dist(csr_subgraph, 0))], axis=1)
        ind_labels = th.tensor([[0, 1], [1, 0]])
        node_labels = th.cat([ind_labels, dists]) if dists.size() else ind_labels

        # 3. prune nodes that are at a distance greater than hop from neigh of the target nodes
        pruned_mask = th.max(node_labels, axis=1)[0] <= hop
        nodes, node_labels = nodes[pruned_mask], node_labels[pruned_mask]
    else:
        raise NotImplementedError
    return nodes, node_labels

# @profile
def subgraph_extraction_labeling(ind, graph, mode="bipartite", 
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels(ind, graph, mode, 
                                                   hop, sample_ratio, max_nodes_per_hop)

    subgraph = graph.subgraph(nodes)

    if mode == "bipartite":
        subgraph.ndata['nlabel'] = one_hot(node_labels, (hop+1)*2)
    elif mode == "homo":
        subgraph.ndata['nlabel'] = one_hot(node_labels, hop+1)
    elif mode == "grail":
        subgraph.ndata['nlabel'] = th.cat([one_hot(node_labels[:, 0], hop+1), 
                                    one_hot(node_labels[:, 1], hop+1)], dim=1)
    else:
        raise NotImplementedError
    # subgraph.ndata['x'] = th.cat([subgraph.ndata['nlabel'], subgraph.ndata['refex']], dim=1)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # refex_feature = extract_refex_feature(subgraph).to(th.float)
    # subgraph.ndata['x'] = th.cat([subgraph.ndata['x'], refex_feature], dim=1)

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges())
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[0]]
    sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[1]]
    _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
    subgraph.edata['edge_mask'][target_edges] = 0

    return subgraph

if __name__ == "__main__":
    import time
    from data import MovieLens
    movielens = MovieLens("ml-100k", testing=True)

    train_edges = movielens.train_rating_pairs
    train_graph = movielens.train_graph

    idx = 0
    u, v = train_edges[0][idx], train_edges[1][idx]
    subgraph = subgraph_extraction_labeling(
                    (u, v), train_graph, 
                    hop=1, sample_ratio=1.0, max_nodes_per_hop=200)
    # t_start = time.time()
    # refex_feature = extract_refex_feature(train_graph).to(th.float)
    # print("Epoch time={:.2f}".format(time.time()-t_start))
    pass
