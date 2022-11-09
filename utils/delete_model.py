import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('root',type=str)
args = parser.parse_args()
root = args.root

for rootpth,_dirs,files in os.walk(os.path.join(root)):
    if len(files) > 0:
        for file in files:
            if file in ['checkpoint'] or file.startswith('model.pth.'):
                os.remove(os.path.join(rootpth,file))
                print('Done! remove file: %s'%os.path.join(rootpth,file))