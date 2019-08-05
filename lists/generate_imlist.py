
import os

num_bgs_train = 100
num_bgs_test = 20
train_data_file = 'train.txt'
test_data_file = 'test.txt'

train_file = '/media/hao/DATA/Combined_Dataset/Training_set/training_fg_names.txt'
train_file_bg = '/media/hao/DATA/Combined_Dataset/Training_set/training_bg_names.txt'
test_file = '/media/hao/DATA/Combined_Dataset/Test_set/test_fg_names.txt'
fg_names = [name for name in open(train_file).read().splitlines()]
bg_names = [name for name in open(train_file_bg).read().splitlines()]
fg_names_test = [name for name in open(test_file).read().splitlines()]

image_head = 'merged'
alpha_head = 'alpha'
trimap_head = 'trimaps'
fg_head = 'fg'
bg_head = 'train2014'

def write_datalist(img_name, bg_names, idx, f):
    prefix = 'Training_set'
    img_path = os.path.join(prefix, image_head, img_name+'_'+str(idx)+'.png')
    msk_path = os.path.join(prefix, alpha_head, img_name+'.jpg')
    fg_path = os.path.join(prefix, fg_head, img_name+'.jpg')
    bg_path = os.path.join(prefix, bg_head, bg_names)
    f.write(img_path+'\t'+msk_path+'\t'+fg_path+'\t'+bg_path+'\n')

def write_datalist_test(img_name, idx, f):
    prefix = 'Test_set'
    img_path = os.path.join(prefix, image_head, img_name+'_'+str(idx)+'.png')
    msk_path = os.path.join(prefix, alpha_head, img_name+'.png')
    trimap_path = os.path.join(prefix, trimap_head, img_name+'_'+str(idx)+'.png')
    f.write(img_path+'\t'+msk_path+'\t'+trimap_path+'\n')

if __name__ == '__main__':
    with open(train_data_file, 'w') as f:
        count = 0
        for name in fg_names:
            img_name, ext = os.path.splitext(name)
            for idx in range(num_bgs_train):
                write_datalist(img_name, bg_names[count], idx, f)
                count += 1
    
    with open(test_data_file, 'w') as f:
        for name in fg_names_test:
            img_name, ext = os.path.splitext(name)
            for idx in range(num_bgs_test):
                write_datalist_test(img_name, idx, f)