import argparse
import csv
import os


def dataset_statistic(dataset_path: str, with_labels: bool = False) -> dict:
    '''
    Statistic KG dataset.
        Dataset's file names:
        - train.txt
        - valid.txt (unlabeled)
        - test.txt (unlabeled)
        - valid_w_label.txt (labeled, if with_labels=True)
        - test_w_label.txt (labeled, if with_labels=True).

    Return
        Columns (with_labels=False):
        + Dataset: dataset name
        + Entities: number of entities
        + Relations: number of relations
        + Triples: number of triples
        + Training: number of training triples
        + Validation: number of validation triples
        + Test: number of test triples

        Columns (with_labels=True):
        + Dataset: dataset name
        + Entities: number of entities
        + Relations: number of relations
        + Triples: number of triples
        + Training: number of training triples
        + Unlabeled Validation: number of unlabeled validation triples
        + Unlabeled Test: number of unlabeled test triples
        + Labeled Validation: number of labeled validation triples
        + Labeled Test: number of labeled test triples
    '''
    dataset_name = os.path.basename(dataset_path).upper()
    nentity = 0
    nrelation = 0
    ntriples = 0
    ntraining = 0
    nunlabeled_validation = 0
    nunlabeled_test = 0
    nlabeled_validation = 0
    nlabeled_test = 0

    with open(os.path.join(dataset_path, 'entities.dict')) as fin:
        for line in fin:
            nentity += 1
    
    with open(os.path.join(dataset_path, 'relations.dict')) as fin:
        for line in fin:
            nrelation += 1
    
    with open(os.path.join(dataset_path, 'train.txt')) as fin:
        for line in fin:
            ntraining += 1
    
    with open(os.path.join(dataset_path, 'valid.txt')) as fin:
        for line in fin:
            nunlabeled_validation += 1
    
    with open(os.path.join(dataset_path, 'test.txt')) as fin:
        for line in fin:
            nunlabeled_test += 1
    
    ntriples = ntraining + nunlabeled_validation + nunlabeled_test

    if with_labels:
        with open(os.path.join(dataset_path, 'valid_w_label.txt')) as fin:
            for line in fin:
                nlabeled_validation += 1
    else:
        nlabeled_validation = 0

    if with_labels:
        with open(os.path.join(dataset_path, 'test_w_label.txt')) as fin:
            for line in fin:
                nlabeled_test += 1
    else:
        nlabeled_test = 0

    return {
        'Dataset': dataset_name,
        'Entities': nentity,
        'Relations': nrelation,
        'Triples': ntriples,
        'Training': ntraining,
        'Unlabeled Validation': nunlabeled_validation,
        'Unlabeled Test': nunlabeled_test,
        'Labeled Validation': nlabeled_validation,
        'Labeled Test': nlabeled_test,
    }


if __name__ == '__main__':
    # Pass dataset_path by command line argument, default is the first argument
    # dataset_path: data/FB15k-237, data/FB15k, data/wn18rr, data/wn18, data/hetionet
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, default='data/FB15k-237', help='Dataset path')
    args = parser.parse_args()
    dataset_path = args.dataset_path

    if dataset_path in ['data/FB15k-237', 'data/wn18rr', 'data/hetionet']:
        with_labels = True
    else:
        with_labels = False
    print(dataset_statistic(dataset_path, with_labels=with_labels))

    dataset_name = os.path.basename(dataset_path)
    output_path = f'visualization/outputs/dataset_statistic/{dataset_name}_statistic.csv'
    # Print the statistic result to a CSV file
    with open(output_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(dataset_statistic(dataset_path, with_labels=with_labels).keys())
        writer.writerow(dataset_statistic(dataset_path, with_labels=with_labels).values())
    print(f'Statistic result saved to {output_path}')