"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from data.BMDataset import BMDataset
from torch.utils.data import Dataset, DataLoader


class BMBags(Dataset):
    def __init__(self, dataset, target_number=1, bag_type = 1, mean_bag_length=3, var_bag_length=1, num_bag=250, seed=1, train=True):
        self.dataset = dataset
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_dataset = len(self.dataset)
        self.num_bag = self.num_dataset//self.mean_bag_length
        self.bag_type = bag_type

        #print('Number of instances is {}'.format(self.num_dataset))
        #print('Number of bags is {}'.format(self.num_bag))
        self.r = np.random.RandomState(seed)

        if self.bag_type == 0:  # normal distribution
            self.bags_list, self.labels_list = self._create_bags()
        elif self.bag_type == 1: # equal bags
            self.bags_list, self.labels_list = self.equal_bags()
    ## patient bag is still under test
    def patient_bags(self):  # each bag matches a patient rather rather random instances
        loader = data_utils.DataLoader(self.dataset, batch_size=self.num_dataset, shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []
        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.num_dataset, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list
     
    def equal_bags(self):   # each bag is with equal length
        loader = data_utils.DataLoader(self.dataset, batch_size=self.num_dataset, shuffle=False) # BM dataset all in
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []
        for i in range(self.num_bag):
            bag_length = self.mean_bag_length
            indices = torch.LongTensor(self.r.randint(0, self.num_dataset, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list


    def _create_bags(self):
        loader = data_utils.DataLoader(self.dataset, batch_size=self.num_dataset, shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []
        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.num_dataset, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = max(self.labels_list[index])
        return bag, label


if __name__ == "__main__":
  ### rewrite this part to test bag.py
    
    # debug code to watch how bags are created
    data_root = '/media/hhy/data/USdata/MergePhase1/test_0.3'

    train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])


    trainset = BMDataset(data_root + '/train', train_transform)
    trainbag = BMBags(dataset = trainset, bag_type=0)

    testset = BMDataset(data_root+'/test',test_transform)
    testbag = BMBags(dataset=testset, bag_type=0)
    
    bag_state = []
    instance_num = []
    for i in range(len(trainbag)):
        state = trainbag[i][1].int().tolist()
        bag_state.append(state)
    
    positive = bag_state.count(1)
    print ('There are {} positive bags out of {} bags'.format(positive, len(trainbag)))
    
    bag_state = []
    instance_num = []
    for i in range(len(testbag)):
        state = testbag[i][1].int().tolist()
        bag_state.append(state)
    
    positive = bag_state.count(1)
    print ('There are {} positive bags out of {} bags'.format(positive, len(testbag)))
    
    
    #testbag = BMBags(dataset = testset)
    '''
    train_loader = data_utils.DataLoader(BMBags(BMDataset),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(BMBags(BMDataset),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
    '''