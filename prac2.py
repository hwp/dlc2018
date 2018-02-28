#!/usr/bin/env python

import argparse

import torch
import dlc_practical_prologue as prologue

def nearest_classification(train_input, train_target, x):
    mdis, midx = torch.min(torch.sum(torch.pow(train_input - x, 2.0), dim=1), dim=0)
    return train_target[midx][0]

def compute_nb_errors(train_input, train_target, test_input, test_target,
                      mean=None, proj=None):
    if mean is not None:
        train_input_proj = train_input - mean
        test_input_proj = test_input - mean
    else:
        train_input_proj = train_input
        test_input_proj = test_input

    if proj is not None:
        train_input_proj = torch.mm(train_input_proj, proj.transpose(0, 1))
        test_input_proj = torch.mm(test_input_proj, proj.transpose(0, 1))

    pred = [nearest_classification(train_input_proj, train_target, x)
                                                    for x in test_input_proj]

    return torch.sum(torch.ne(torch.LongTensor(pred), test_target))

def PCA(x):
    mean = torch.mean(x, dim=0)
    x = x - mean
    e, v = torch.eig(torch.mm(x.transpose(0, 1), x), eigenvectors=True)
    e = e[:,0]              # take real part
    proj = v[:, torch.sort(e, descending=True)[1]].transpose(0, 1)

    # ATTENTION: each row of proj is an eigenvector
    return mean, proj

if __name__ == '__main__':
    # parse arguments
    parser = prologue.parser
    parser.add_argument('pca_dim', metavar='PCA_DIM', type=int,
                        help='dimension of PCA subspace')
    args = parser.parse_args()
    pca_dim = args.pca_dim


    # load data
    prologue.init(args)
    train_input, train_target, test_input, test_target = prologue.load_data(args)
    print 'data loaded'

    # compute transform
    mean, proj = PCA(train_input)
    proj = proj[:pca_dim]
    print 'transform computed'

    # test
    err = compute_nb_errors(train_input, train_target, test_input, test_target,
                            mean=mean, proj=proj)
    print '# of Error : %d' % err

