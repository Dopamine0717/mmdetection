import os
import argparse

def check_file_nums(dir_path, recursion=False):
    '''This function can help us check the file numbers of the dir_path.
    Args:
        dir_path:{str} The path of the folder to check.
        recursion:{bool} whether to preform recursive query. Default is False.

    Returns: numbers of file, numbers of dir
    '''
    if os.path.isdir(dir_path):
        if not recursion:
            filelist = [i for i in os.listdir(dir_path) if not os.path.isdir(os.path.join(dir_path,i))]
            return len(filelist), 0
        else:
            filelist = os.listdir(dir_path)
            file_nums = 0
            dir_nums = 0
            for i in filelist:
                if os.path.isdir(os.path.join(dir_path,i)):
                    dir_nums += 1
                    file_num, dir_num = check_file_nums(os.path.join(dir_path,i), recursion)
                    file_nums += file_num
                    dir_nums += dir_num
                else:
                    file_nums += 1
            return file_nums, dir_nums
    else:
        print("The path you entered is not a folder!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('check the number of files')
    parser.add_argument(
        "--dir-path", 
        type=str, 
        default="/shared/xjd/DataSets/transmission_line_detection/train14000/Annotations", 
        help="dir path"
        )
    args = parser.parse_args()
    file_nums, dir_nums = check_file_nums(args.dir_path, recursion=True)
    print(f"The number of file is {file_nums}")
    print(f"The number of dir is {dir_nums}")