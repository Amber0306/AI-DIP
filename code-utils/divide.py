import os
from shutil import copyfile
from tokenize import String


def check_path(origin_path, base_path):
    if not os.path.exists(origin_path):
        return -1
    base_images = origin_path+'\\'+'JPEGImages'
    base_ann = origin_path+'\\'+'Annotations'
    if not os.path.exists(base_images):
        return -1
    if not os.path.exists(base_ann):
        return -1   

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    os.mkdir(base_path+'\\'+'train')
    os.mkdir(base_path+'\\'+'val')
    os.mkdir(base_path+'\\''test')
    train_images = base_path+'\\'+'train\\images'
    train_ann = base_path+'\\'+'train\\annotations'
    os.mkdir(train_images)
    os.mkdir(train_ann)
    val_images = base_path+'\\'+'val\\images'
    val_ann = base_path+'\\'+'val\\annotations'
    os.mkdir(val_images)
    os.mkdir(val_ann)
    test_images = base_path+'\\'+'test\\images'
    test_ann = base_path+'\\'+'test\\annotations'
    os.mkdir(test_images)
    os.mkdir(test_ann)
    path_list = [base_images,base_ann,train_images,train_ann,val_images, val_ann, test_images, test_ann]
    return path_list

def divideOne(index:int,src_images:String,src_ann:String, dst_images:String, dst_ann:String):
    if index <0 or index >9:
        return 'error'
    image_files= os.listdir(src_images)
    for file in image_files: 
        temp = int(file[-5])
        if temp == index and not os.path.isdir(file): 
          src = src_images+'/'+file
          dst = dst_images+'/'+file
          copyfile(src,dst)

    ann_files= os.listdir(src_images)     
    for file in ann_files: 
        temp = int(file[-5])
        if temp == index and not os.path.isdir(file): 
          src = src_images+'/'+file
          dst = dst_images+'/'+file
          copyfile(src,dst)
    return

def divide(path_list:list, train_part:list, val_part:list, test_part:list):
    
    # for i in train_part:
    #     divideOne(train_part[i],path_list[0],path_list[1],path_list[2],path_list[3])
    # for i in train_part:
    #     divideOne(val_part[i],path_list[0],path_list[1],path_list[4],path_list[5])
    # for i in train_part:
    #     divideOne(test_part[i],path_list[0],path_list[1],path_list[6],path_list[7])

    image_files= os.listdir(path_list[0])
    for file in image_files: 
        temp = int(file[-5])
        src = path_list[0]+'/'+file
        if os.path.isdir(file):
            continue
        dst=''
        if temp in train_part:
            dst = path_list[2]+'/'+file
        elif temp in val_part:
            dst = path_list[4]+'/'+file 
        elif temp in test_part:
            dst = path_list[6]+'/'+file
        else:
            continue
        copyfile(src,dst)

    ann_files= os.listdir(path_list[1])     
    for file in ann_files: 
        temp = int(file[-5])
        src = path_list[1]+'/'+file
        if os.path.isdir(file):
            continue
        dst=''
        if temp in train_part:
            dst = path_list[3]+'/'+file
        elif temp in val_part:
            dst = path_list[5]+'/'+file 
        elif temp in test_part:
            dst = path_list[7]+'/'+file
        else:
            continue
        copyfile(src,dst)

    return



if __name__ == '__main__':
    origin_path = 'E:\研一\数据集\SSDD\SSDD'
    base_path = 'E:\研一\数据集\SSDD\SSDD1'
    path_list = check_path(origin_path, base_path)
    train_part = [2,3,4,5,6,7,8]
    val_part = [0]
    test_part = [1,9]
    divide(path_list, train_part,val_part, test_part)
