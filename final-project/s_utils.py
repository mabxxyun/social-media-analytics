import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="File name of dataset.",
        default="tvshow",
    )
    
    #parser.add_argument("-a", "--all", help="noPCA",action='store_true')
    parser.add_argument("-j", "--jaccard", help="jaccard",action='store_true')
    parser.add_argument("-c", "--consine", help="consine similarity",action='store_true')
    parser.add_argument("-t", "--shortest", help="shortest path",action='store_true')
    parser.add_argument("-i", "--same_community", help="same community",action='store_true')
    parser.add_argument("-b", "--common_neighbor", help="common neighbor",action='store_true')
    parser.add_argument("-a", "--adamic", help="adamic adar",action='store_true')
    parser.add_argument("-u", "--resource", help="resource allocation index",action='store_true')
    parser.add_argument("-f", "--preferential", help="preferential attachment",action='store_true')
    parser.add_argument("-y", "--svd", help="svd",action='store_true')
    
    parser.add_argument("-e", "--gemsec", help="Gemsec", action='store_true')
    parser.add_argument("-s", "--gemsecReg", help="GemsecWithRegularization", action='store_true')
    parser.add_argument("-l", "--deepwalk", help="Deepwalk", action='store_true')
    parser.add_argument("-k", "--deepwalkReg",  help="DeepwalkWithRegularization", action='store_true')
    parser.add_argument("-x", "--n2v",  help="DeepwalkWithRegularization", action='store_true')
    
    parser.add_argument("--dim", help="Dimension of High Dim Embedding", type=float, default=16)
    
    parser.add_argument("-p", "--p_gemsec", help="Gemsec with PCA", action='store_true')
    parser.add_argument("-m", "--p_gemsecReg", help="GemsecWithRegularization with PCA", action='store_true')
    parser.add_argument("-d", "--p_deepwalk", help="Deepwalk with PCA", action='store_true')
    parser.add_argument("-w", "--p_deepwalkReg",  help="DeepwalkWithRegularization with PCA", action='store_true')
    parser.add_argument("-n", "--p_n2v", help="node2vec with PCA",action='store_true')
    
    parser.add_argument("-v", "--visualize", action='store_true')
    parser.add_argument("-r", "--randomforest", action='store_true')
    parser.add_argument("-g", "--gradientboost", action='store_true')

    args = parser.parse_args()
    return args
