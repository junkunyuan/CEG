import os
import argparse
import json


def split_accs(path):
    accs = {}
    targets = []
    with open(path, encoding='utf-8') as f:
        contents = f.readlines()
        for line in contents:
            if line.startswith('target_domains'):
                target = eval(line[16:])[0]
                if target not in targets:
                    targets.append(target)
            if line.startswith('seed'):
                seed = '_'.join(['seed', line[6:-1]])
            if line.startswith('  NUM_LABELED:'):
                num_labeled = '_'.join(['labeled_num', line[15:-1]])

                accs.setdefault(num_labeled, {})
                accs[num_labeled].setdefault(seed, {})
                accs[num_labeled][seed].setdefault(target, [])

            if line.startswith('* accuracy: '):
                acc = line[12:-2]
                accs[num_labeled][seed][target].append(float(acc))
    return accs, targets


def save_accs_csv(accs, targets, root, name='accs'):
    path = os.path.join(root, name+'.csv')
    recorders = []
    with open(path, 'w') as f:
        for lnum, v_lnum in accs.items():
            print(lnum, file=f)
            print(','.join(targets), file=f)
            for seed, v_seed in v_lnum.items():
                for _, v_target in v_seed.items():
                    if len(v_target) > 0:
                        recorders.append(v_target[-1])
                        print(v_target[-1], end=',', file=f)
                print(seed, file=f)
            print(file=f)


def save_accs_process(accs, root, name='accs'):
    path = os.path.join(root, name+'.json')
    with open(path, 'w') as f:
        json.dump(accs, f)


def main(config, log_name, save_name):

    path = os.path.join(config['root'], log_name)
    accs, targets = split_accs(path)

    save_accs_csv(accs, targets, config['save_root'], name=save_name.split('.')[0])
    print('{} Done! Save accuracy number to a .csv file.'.format(log_name))

    if config['record_acc']:
        save_accs_process(accs, config['save_root'], name=save_name)
    print('{} Done! Save all accuracy number to a json file.'.format(log_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='The log file root dir.')
    parser.add_argument('--save_root', type=str, default='', help='The save file root dir.')
    parser.add_argument('--log_name', type=str, default='out.txt', help='The log file name.')
    parser.add_argument('--save_name', type=str, default='', help='The save acc file name.')
    parser.add_argument('--all', action='store_true', help='Analysis all log files in the root dir')
    parser.add_argument('--record_acc', action='store_true', help='Record all acc data.')
    parser.add_argument('--delete',action='store_true',help='Delet all .csv file in root dir')
    args = parser.parse_args()

    root = args.root
    save_root = args.root if args.save_root == '' else args.save_root
    log_names = os.listdir(root) if args.all else [args.log_name]
    save_names = log_names if args.save_name == '' else [args.save_name]
    record_acc = args.record_acc
    delete = args.delete

    config = {
        'root': root,
        'save_root': save_root,
        'record_acc': record_acc,
        'delete':delete
    }
    for log_name, save_name in zip(log_names, save_names):
        splitext = os.path.splitext(log_name)
        if delete == True:
            if splitext[-1] == '.csv':
                path = os.path.join(root,log_name)
                os.remove(path)
                print('Done! remove file:{}'.format(path))
        else:
            if splitext[-1] == '.txt':
                main(config, log_name, save_name)
        
        
