import os
from FilesAccumulator import FilesAccumulator


def write_file(input, filename):
    print(filename)
    with open(filename,'w') as fn:
        print('#!/bin/bash',file=fn)
        print('#SBATCH -o detection.err%j',file=fn)
        print('#SBATCH -t 6:00:00',file=fn)
        print('#SBATCH -N 1 -n 4',file=fn)
        print('#SBATCH -p gpu',file=fn)
        print('#SBATCH --gres=gpu:1',file=fn)
        print('#SBATCH -A shah',file=fn)
        print('\n',file=fn)
        print('cd /brazos/shah/pranav/apd',file=fn)

        print('module load python/3.7', file=fn)
        print('pip install -r working.txt', file=fn)
        cmdf = 'python demo.py ' + '-f ' + input
        print(cmdf,file=fn)

input_folder = "/home/pmantini/Downloads/SVFPD"
files = FilesAccumulator(input_folder)
files_list = files.find([".mkv", ".avi"], [])
output_folder = "scripts/"
count = 0
for k in files_list:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    input = k.rsplit("SVFPD", 1)[1]

    input_file = "SVFPD"+input
    ouput_file = output_folder + "apd_%s.sh" % count
    write_file(input_file, ouput_file)
    break
    # count += 1
