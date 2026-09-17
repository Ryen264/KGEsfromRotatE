import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from code.model import KGEModel, resolve_rotate_score_mode
from code.loss import UniGammaController, build_training_optimizer, is_learnable_kgau_gammas, set_optimizer_learning_rates, update_kgau_gamma_schedule
from code.strategy import get_strategy, resolve_strategy_name

def steps_per_epoch(num_train_triples, batch_size):
    batches = (num_train_triples + batch_size - 1) // batch_size
    return 2 * batches

def parse_args(args=None):
    # =========================================================================
    # PASS 1: Quét tìm file config JSON trước
    # =========================================================================
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        'config_file', nargs='?', type=str, default=None, 
        help='Đường dẫn đến file cấu hình JSON (Ví dụ: configs/ComplEx128_WN18RR_kgau4_100e.json)'
    )
    
    known_args, remaining_argv = conf_parser.parse_known_args(args)
    
    defaults = {}
    if known_args.config_file and known_args.config_file.endswith('.json'):
        with open(known_args.config_file, 'r') as fjson:
            defaults = json.load(fjson)

    # =========================================================================
    # PASS 2: Khởi tạo parser chính và nạp thông số
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        parents=[conf_parser] # Kế thừa lại tham số config_file
    )

    parser.add_argument('--no_cuda', action='store_true', help='Ép không dùng GPU (Mặc định tự động dùng GPU nếu có)')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, help='Region Id for Countries S1/S2/S3 datasets')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('--negative_chunk_size', default=256, type=int, help='Max negatives scored per chunk')
    parser.add_argument('-d', '--dim', default=500, type=int)
    parser.add_argument('-g', '--margin_gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    
    parser.add_argument('--strategy', default=None, type=str,
                        choices=['uniform', 'bernoulli', 'selfadv', 'kbgan', '1vsall', 'kvsall', 'kgau'],
                        help='Training strategy. Include "kbgan" for adversarial sampling via Generator.')
    
    # Args cho Value / Triple Classification
    parser.add_argument('--triple_classification', action='store_true', 
                        help='Evaluate on Value / Triple Classification task')
    parser.add_argument('--threshold_mode', default='relation', type=str, choices=['global', 'relation'],
                        help='Threshold search mode for Triple Classification')
    parser.add_argument('--generator_checkpoint', default=None, type=str,
                        help='Path to the pre-trained KGE generator checkpoint (required if strategy=kbgan)')

    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization_coeff', default=0.0, type=float)
    parser.add_argument('-rp', '--regularization_p', default=3, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--uni_weight', action='store_true')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=4, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warm_up_epochs', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int)
    parser.add_argument('--test_log_steps', default=1000, type=int)
    
    parser.add_argument('--loss', default='sans', type=str,
                        choices=['se', 'hinge', 'bce', 'mr', 'bpr', 'ce', 'sans', 'kgau', 'kgmau', 'kgmamu'])

    parser.add_argument('--align_margin', default=0.0, type=float)
    parser.add_argument('--align_alpha', default=2.0, type=float)
    parser.add_argument('--uniform_margin', default=2.0, type=float)
    parser.add_argument('--uniform_t', default=4, type=float)
    parser.add_argument('--uniform_pair_chunk_size', default=256, type=int)
    parser.add_argument('--uniform-gamma-q', dest='uniform_gamma_q', default=1.0, type=float)
    parser.add_argument('--uniform-gamma-y', dest='uniform_gamma_y', default=1.0, type=float)
    parser.add_argument('--uniform-gamma-e', dest='uniform_gamma_e', default=0.0, type=float)

    parser.add_argument('--learnable_kgau_gammas', action='store_true')
    parser.add_argument('--log_kgau_gamma_lr', default=None, type=float)
    parser.add_argument('--gamma_linear_schedule', action='store_true')
    parser.add_argument('--gamma_schedule_end', default=0.1, type=float)
    parser.add_argument('--gamma_schedule_start_epoch', default=0, type=int)
    parser.add_argument('--gamma_schedule_epochs', default=0, type=int)
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    # =========================================================================
    # Áp dụng Defaults từ JSON và Parse CLI args để đè lên
    # =========================================================================
    parser.set_defaults(**defaults)
    parsed_args = parser.parse_args(remaining_argv)
    
    # 1. Tự động nhận diện GPU: Luôn dùng cuda trừ khi cờ --no_cuda được bật
    if not parsed_args.no_cuda:
        parsed_args.cuda = torch.cuda.is_available()
    else:
        parsed_args.cuda = False
        
    # 2. Tự động thiết lập quy trình: Nếu chạy bằng config, tự kích hoạt train/valid/test
    if known_args.config_file:
        if not (parsed_args.do_train or parsed_args.do_valid or parsed_args.do_test):
            parsed_args.do_train = True
            parsed_args.do_valid = True
            parsed_args.do_test = True
            
    # 3. Tự động gán save_path vào thư mục outputs/ với timestamp
    if parsed_args.do_train and parsed_args.save_path is None:
        if known_args.config_file:
            config_name = os.path.basename(known_args.config_file).replace('.json', '')
        else:
            config_name = "main_cli"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        parsed_args.save_path = os.path.join('outputs', f"{config_name}_{timestamp}")
            
    return parsed_args


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


def save_model(model, optimizer, save_variable_list, args, checkpoint_name='checkpoint'):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate.
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson, indent=4)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, checkpoint_name)
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


def clone_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


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
    Write logs to checkpoint file ONLY (to avoid breaking tqdm progress bar on console)
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs to File
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def save_results_to_txt(filepath, mode, metrics):
    '''
    Save evaluation metrics to results.txt
    '''
    with open(filepath, 'a') as f:
        f.write(f"=== {mode} Results ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")


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
    
    # Write logs to file only
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
    
    print('Model:', args.model)
    print('Data Path:', args.data_path)
    print('#entity:', nentity)
    print('#relation:', nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

    steps_per_epoch_val = steps_per_epoch(len(train_triples), args.batch_size)
    max_steps_internal = args.epochs * steps_per_epoch_val
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        dim=args.dim,
        margin_gamma=args.margin_gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        score_mode=resolve_rotate_score_mode(args) if args.model == 'RotatE' else 'distance',
    )
    
    if args.cuda:
        kge_model = kge_model.cuda()

    # --- KHỞI TẠO GENERATOR CHO KBGAN ---
    if resolve_strategy_name(args) == 'kbgan':
        if not args.generator_checkpoint:
            raise ValueError('KBGAN strategy requires a pre-trained generator. Please provide --generator_checkpoint')
        
        print('Loading Generator for KBGAN from %s...' % args.generator_checkpoint)
        gen_checkpoint = torch.load(os.path.join(args.generator_checkpoint, 'checkpoint'))
        
        with open(os.path.join(args.generator_checkpoint, 'config.json'), 'r') as fjson:
            gen_args = json.load(fjson)
            
        generator_model = KGEModel(
            model_name=gen_args['model'],
            nentity=nentity,
            nrelation=nrelation,
            dim=gen_args['dim'],
            margin_gamma=gen_args['margin_gamma'],
            double_entity_embedding=gen_args['double_entity_embedding'],
            double_relation_embedding=gen_args['double_relation_embedding'],
            score_mode=gen_args.get('score_mode', 'distance')
        )
        generator_model.load_state_dict(gen_checkpoint['model_state_dict'], strict=False)
        generator_model.eval() 
        if args.cuda:
            generator_model = generator_model.cuda()
            
        kge_model.generator = generator_model
    # ------------------------------------

    init_step = 0
    if args.init_checkpoint:
        print('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print('Randomly Initializing %s Model...' % args.model)

    if is_learnable_kgau_gammas(args):
        UniGammaController(args).ensure_model_params(kge_model)
    
    if args.do_train:
        strategy = get_strategy(args)
        train_iterator = strategy.build_train_iterator(train_triples, nentity, nrelation)
        
        current_learning_rate = args.learning_rate
        optimizer = build_training_optimizer(kge_model, args)
        warm_up_epochs_val = (args.warm_up_epochs if args.warm_up_epochs is not None else args.epochs // 2)
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
    
    if args.do_train:
        print('\n--- Start Training ---')
        print(f"Output directory: {args.save_path}")
        print(f"Strategy: {resolve_strategy_name(args)} | Loss: {args.loss} | Epochs: {args.epochs}")
        
        training_logs = []
        best_valid_metric = float('-inf')
        best_valid_epoch = None
        best_model_state = None
        valid_metric_key = 'MRR' if not args.triple_classification else 'f1'
        
        init_epoch = init_step // steps_per_epoch_val + 1
        
        # Vòng lặp ngoài theo Epochs
        epoch_bar = tqdm(range(init_epoch, args.epochs + 1), desc='Training', unit='epoch', dynamic_ncols=True)
        
        for epoch in epoch_bar:
            if is_learnable_kgau_gammas(args):
                args.current_epoch = epoch
                update_kgau_gamma_schedule(args)
            
            # Vòng lặp trong theo Batch/Steps
            step_bar = tqdm(range(steps_per_epoch_val), desc=f'  Epoch {epoch}', unit='batch', leave=False, dynamic_ncols=True)
            
            for _ in step_bar:
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
                training_logs.append(log)
                step += 1
                
                # --- Điều chỉnh Learning Rate ---
                if step >= warm_up_steps_internal:
                    current_learning_rate = current_learning_rate / 10
                    tqdm.write(f'Change learning_rate to {current_learning_rate} at epoch {epoch}')
                    if is_learnable_kgau_gammas(args) and len(optimizer.param_groups) > 1:
                        set_optimizer_learning_rates(
                            optimizer, current_learning_rate,
                            getattr(args, 'log_kgau_gamma_lr', None) or current_learning_rate,
                        )
                    else:
                        for group in optimizer.param_groups:
                            group['lr'] = current_learning_rate
                    warm_up_steps_internal = warm_up_steps_internal * 3
                    warm_up_epochs_val = warm_up_epochs_val * 3
                
                # --- Ghi Log Train ---
                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([l[metric] for l in training_logs])/len(training_logs)
                    
                    postfix_str = ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
                    step_bar.set_postfix_str(postfix_str)
                    log_metrics('Training average', step, metrics)
                    training_logs = []

            # --- LƯU TRỮ VÀ ĐÁNH GIÁ CUỐI MỖI EPOCH ---
            # Lưu checkpoint cuối mỗi epoch thay vì theo step
            save_variable_list = {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_epochs': warm_up_epochs_val
            }
            save_model(kge_model, optimizer, save_variable_list, args)
            
            # Đánh giá Validation cuối mỗi Epoch
            if args.do_valid:
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                
                # Cập nhật thông số lên thanh Epoch
                epoch_bar.set_postfix_str(f"Valid {valid_metric_key}={metrics.get(valid_metric_key, 0):.4f}")
                
                if valid_metric_key in metrics and metrics[valid_metric_key] > best_valid_metric:
                    best_valid_metric = metrics[valid_metric_key]
                    best_valid_epoch = epoch
                    best_model_state = clone_model_state(kge_model)
                    
                    save_variable_list.update({
                        'best_valid_metric': best_valid_metric,
                        'best_valid_metric_name': valid_metric_key,
                        'best_valid_epoch': best_valid_epoch,
                    })
                    save_model(
                        kge_model, optimizer, save_variable_list, args,
                        checkpoint_name='checkpoint_best',
                    )
                    tqdm.write(f'---> New best valid {valid_metric_key} = {best_valid_metric:.4f} at epoch {epoch}')

        # Sau khi kết thúc huấn luyện, nạp lại mô hình tốt nhất
        if best_model_state is not None:
            kge_model.load_state_dict(best_model_state)
            print(f'\nRestored best model at epoch {best_valid_epoch} (valid {valid_metric_key} = {best_valid_metric:.4f}) for final evaluation')

    # =========================================================================
    # Giai đoạn đánh giá cuối cùng & Lưu kết quả ra file results.txt
    # =========================================================================
    results_file = os.path.join(args.save_path or args.init_checkpoint, 'results.txt')

    if args.do_valid:
        print('\nFinal Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        save_results_to_txt(results_file, 'Valid', metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    if args.do_test:
        print('\nFinal Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        save_results_to_txt(results_file, 'Test', metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    if args.evaluate_train:
        print('\nFinal Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Train', step, metrics)
        save_results_to_txt(results_file, 'Train', metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    if args.save_path:
        print(f'\nAll results and checkpoints have been saved to: {args.save_path}')

if __name__ == '__main__':
    main(parse_args())