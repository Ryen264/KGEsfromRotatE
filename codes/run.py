from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from loss import UniGammaController, build_training_optimizer, is_learnable_au_gammas, set_optimizer_learning_rates, update_au_gamma_schedule

def steps_per_epoch(num_train_triples, batch_size):
    batches = (num_train_triples + batch_size - 1) // batch_size
    return 2 * batches

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization_coeff', default=0.0, type=float,
                        help='Lp embedding regularization coefficient')
    parser.add_argument('-rp', '--regularization_p', default=3, type=int,
                        help='Lp norm order for embedding regularization')
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs (full head+tail passes over train triples)')
    parser.add_argument('--warm_up_epochs', default=None, type=int,
                        help='Epochs before first learning-rate decay (default: epochs // 2)')
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--loss', default='sans', type=str,
                        choices=['se', 'hinge', 'bce', 'mr', 'bpr', 'ce', 'sans', 'au'],
                        help='Training loss (see codes/loss.py)')

    parser.add_argument('--tuni', default=2, type=float,
                        help='Uniformity temperature for AU loss')
    parser.add_argument('--uni-gamma-query', dest='uni_gamma_query', default=1.0, type=float,
                        help='Initial AU uniformity weight for query embeddings (0=off)')
    parser.add_argument('--uni-gamma-target', dest='uni_gamma_target', default=1.0, type=float,
                        help='Initial AU uniformity weight for target embeddings (0=off)')
    parser.add_argument('--uni-gamma-entity', dest='uni_gamma_entity', default=0.0, type=float,
                        help='Initial AU uniformity weight for entity embeddings (0=off)')
    parser.add_argument('--uni-gamma-head', dest='uni_gamma_head', default=0.0, type=float,
                        help='Initial AU uniformity weight for head entity embeddings (0=off)')
    parser.add_argument('--uni-gamma-tail', dest='uni_gamma_tail', default=0.0, type=float,
                        help='Initial AU uniformity weight for tail entity embeddings (0=off)')
    parser.add_argument('--uni-gamma-relation', dest='uni_gamma_relation', default=0.0, type=float,
                        help='Initial AU uniformity weight for relation embeddings (0=off)')
                        
    parser.add_argument('--learnable_au_gammas', action='store_true',
                        help='Learn batch-wise AU gamma down-weighting via log_gamma_adj')
    parser.add_argument('--log_au_gamma_lr', default=None, type=float,
                        help='LR for learnable AU gammas (default: learning_rate)')
    parser.add_argument('--gamma_linear_schedule', action='store_true',
                        help='Linearly anneal AU gamma schedule multiplier over training')
    parser.add_argument('--gamma_schedule_end', default=0.1, type=float,
                        help='Final AU gamma schedule multiplier')
    parser.add_argument('--gamma_schedule_start_epoch', default=0, type=int,
                        help='Epoch when AU gamma schedule starts')
    parser.add_argument('--gamma_schedule_epochs', default=0, type=int,
                        help='AU gamma schedule length (0=full epochs)')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.dim = argparse_dict['dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    steps_per_epoch_val = steps_per_epoch(len(train_triples), args.batch_size)
    max_steps_internal = args.epochs * steps_per_epoch_val
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        dim=args.dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    init_step = 0
    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)

    if is_learnable_au_gammas(args):
        UniGammaController(args).ensure_model_params(kge_model)
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = build_training_optimizer(kge_model, args)
        warm_up_epochs_val = (
            args.warm_up_epochs if args.warm_up_epochs is not None else args.epochs // 2
        )
        warm_up_steps_internal = warm_up_epochs_val * steps_per_epoch_val

        if args.init_checkpoint:
            current_learning_rate = checkpoint['current_learning_rate']
            if 'warm_up_epochs' in checkpoint:
                warm_up_epochs_val = checkpoint['warm_up_epochs']
                warm_up_steps_internal = warm_up_epochs_val * steps_per_epoch_val
            elif 'warm_up_steps' in checkpoint:
                warm_up_steps_internal = checkpoint['warm_up_steps']
                warm_up_epochs_val = warm_up_steps_internal // steps_per_epoch_val
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('epochs = %d' % args.epochs)
    logging.info('steps_per_epoch = %d' % steps_per_epoch_val)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('dim = %d' % args.dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        last_schedule_epoch = None
        
        #Training Loop
        for step in range(init_step, max_steps_internal):
            if is_learnable_au_gammas(args):
                current_epoch = step // steps_per_epoch_val + 1
                if current_epoch != last_schedule_epoch:
                    args.current_epoch = current_epoch
                    update_au_gamma_schedule(args)
                    last_schedule_epoch = current_epoch
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps_internal:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                if is_learnable_au_gammas(args) and len(optimizer.param_groups) > 1:
                    set_optimizer_learning_rates(
                        optimizer, current_learning_rate,
                        getattr(args, 'log_au_gamma_lr', None) or current_learning_rate,
                    )
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = current_learning_rate
                warm_up_steps_internal = warm_up_steps_internal * 3
                warm_up_epochs_val = warm_up_epochs_val * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_epochs': warm_up_epochs_val
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_epochs': warm_up_epochs_val
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
