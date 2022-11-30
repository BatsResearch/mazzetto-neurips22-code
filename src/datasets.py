import os
import copy
import math
import time
from abc import abstractmethod
from collections import defaultdict


import torch
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms


class Data(object):
    def __init__(self, name, folder):
        self.name = name
        self.folder = folder
        
    def get_classes(self):
        print(f'The dataset is composed of {len(self.classes)} classes')
        return self.classes, self.class_idx, self.idx_class, self.classid_imageid
    
    def get_images(self):
        print(f"The dataset contains {len(self.images)} images")
        return self.images, self.image_idx, self.idx_image, self.imageid_class
    
    def get_embeddings(self):
        print("We are loading the feature matrix of all the images..")
        return torch.load(f'{self.folder}{self.name}data/matrix_embs.pt')
    
    def get_attributes(self, aggregate_attributes=None):
        """
        :param aggregate_attributes: e.g., {'dark': [0, 2, 3, 4, 8],
                        'wings': [36, 37]}
        """
        if aggregate_attributes:
            tmp_idx_attr = copy.deepcopy(self.idx_attr)
            for new_att, l in aggregate_attributes.items():
                print(new_att, l)
                if len(l) > 1:
                    to_remove = []
                    for idx, a in tmp_idx_attr.items():
                        if idx in l:
                            to_remove += [idx]
                    for r in to_remove:
                        del tmp_idx_attr[r]
                    tmp_idx_attr[l[0]] = new_att
                else:
                    return('Wrong request for attribute aggregation!')
            
            new_attributes = list(tmp_idx_attr.values())
            self.attr_idx = {attr:idx \
                            for idx, attr in enumerate(new_attributes)}
            self.idx_attr = {idx:attr \
                             for attr, idx in self.attr_idx.items()}
            
            self.old_to_new = {old:self.attr_idx[att] \
                          for old, att in tmp_idx_attr.items()}
            self.new_to_old = {new:old \
                          for old,new in self.old_to_new.items()}
            
            self.new_vectors = {}
            for a, l in aggregate_attributes.items():
                new_idx = self.old_to_new[l[0]]
                merge = np.sum(self.image_attr[:,l], axis=1)
                merge[merge > 0] = 1

                self.new_vectors[new_idx] = merge
                
            new_image_attr = np.zeros((len(self.idx_image), len(self.attr_idx)))
            for new, old in self.new_to_old.items():
                if new in self.new_vectors:
                    new_image_attr[:,new] = self.new_vectors[new]
                else:
                    new_image_attr[:,new] = self.image_attr[:,old]
                    
            self.attributes = new_attributes
            self.image_attr = new_image_attr
            print(f"The dataset contains {len(self.idx_attr)} attributes")
            
            # build the new class-attribute matrix
            self._class_attribute_matrix(continuous=False, agg=True)
            
            return self.attributes, self.attr_idx, self.idx_attr, self.image_attr
        
        else:
            print(f"The dataset contains {len(self.attributes)} attributes")
            return self.attributes, self.attr_idx, self.idx_attr, self.image_attr
        
    def get_seen_unseen_classes(self):
        seen_classes = self._get_classes(self.seen_samples)
        unseen_classes = self._get_classes(self.unseen_samples)
        print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
        return seen_classes, unseen_classes
    
    def get_seen_sample(self):
        print(f"Number of seen examples: {len(self.seen_samples)}")
        return self.seen_samples, self.seen_old_to_new, self.seen_new_to_old
    
    def get_unseen_sample(self):
        print(f"Number of unseen examples: {len(self.unseen_samples)}")
        return self.unseen_samples, self.unseen_old_to_new, self.unseen_new_to_old
        
    def get_class_attribute_matrix(self, continuous=True, bound=None, 
                                   lb=None, ub=None):
        if continuous:
            return self.class_attributes
        else:
            matrix = copy.deepcopy(self.class_attributes)
            if bound:
                matrix[matrix >= bound] = 1
                matrix[matrix < bound] = -1
                return matrix
            
            elif (lb == None) or (ub == None):
                matrix = np.rint(matrix)
                matrix[matrix == 0] = -1
                return matrix
            
            else:
                matrix[matrix >= ub] = 1
                matrix[matrix <= lb] = -1
                matrix[(lb < matrix) & (matrix < ub)] = 0
                
                return matrix
    
    def get_image_attributes(self, idx):
        return self.image_attr[idx]
    
    def get_image_class(self, idx):
        return self.imageid_class[idx]
    
    def get_class_attributes(self, idx):
        return self.class_attributes[idx]
    
    def get_image_embedding(self, idx):
        pass
    
    @abstractmethod
    def _image_indices(self):
        raise NotImplementedError('Subclasses need to implement it!')
    
    @abstractmethod
    def _attribute_indices(self):
        raise NotImplementedError('Subclasses need to implement it!')
        
    @abstractmethod
    def _image_attributes(self):
        raise NotImplementedError('Subclasses need to implement it!')
    
    @abstractmethod
    def _create_splits(self):
        raise NotImplementedError('Subclasses need to implement it!')
        
    def _class_indices(self): 
        """ Defines list of classes, the dictionary (class name, idx),  (idx, class name), 
        and the dictionary (class id, list imgs ids).
        Refer to the notebook datasets to closely look at the variables. 
        """
        
        self.classes = sorted(list(set(list(self.imageid_class.values()))))
        self.class_idx = {cls:idx \
                          for idx, cls in enumerate(self.classes)}
        self.idx_class = {idx:cls \
                          for cls, idx in self.class_idx.items()}
        
        # Match classes to list of images
        self.classid_imageid = defaultdict(list)
        for img, cls in self.imageid_class.items():
            self.classid_imageid[self.class_idx[cls]] += [img]
        
    def _class_attribute_matrix(self, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50, agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
        if continuous:
            print(f"Simply average on continuous attributes, no bounds.")
            for c in self.idx_class:
                self.class_attributes[c] = np.mean(self.image_attr[self.classid_imageid[c]], 
                                          axis=0)
        else:
            if bound:
                print(f"Average on binary attributes, based on bound.")
                for c in self.idx_class:
                    class_img_attr = self.image_attr[self.classid_imageid[c]]
                    class_img_attr[class_img_attr >= bound] = 1
                    class_img_attr[class_img_attr < bound] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
            elif (lb == None) or (ub == None):
                print(f"Average on binary attributes, no bound.")
                for c in self.idx_class:
                    m_round = np.rint(self.image_attr[self.classid_imageid[c]])
                    m_round[m_round == 0] = -1
                    self.class_attributes[c] = np.mean(m_round, axis=0)
            else:
                print(f"Average on binary attributes, based on lb and ub.")
                for c in self.idx_class:
                    class_img_attr = image_attr[classid_imageid[c]]
                    class_img_attr[class_img_attr >= ub] = 1
                    class_img_attr[class_img_attr <= lb] = -1
                    class_img_attr[(lb < class_img_attr) & (class_img_attr < ub)] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
                    
    def _get_classes(self, samples):
        """ Returns set of classes corresponding 
        to the set of samples in input
        """
        classes = []
        for img in samples:
            classes.append(self.imageid_class[img])

        return list(np.unique(classes))
    
    def _generate_embeddings(self, data, path, batch_size=8):
    
        files = os.listdir(f'{path}embeddings/')
        if len(files) == 0:
            print('Computing embeddings..')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = models.resnet101(pretrained=True)
            model.fc = torch.nn.Identity()
            model = model.to(device)
            model.eval()
            
            batching = DataLoader(data, 
                                  batch_size=batch_size, 
                                  shuffle=False)
            
            start = time.time()

            with torch.no_grad():
                for i, (inputs, labels, idx) in enumerate(tqdm(batching)):
                    inputs = inputs.to(device)
                    feat = model(inputs) 
                    # Get the tensor out of the variable
                    features = feat.data 

                    # Save to file
                    torch.save(features, f'{path}embeddings/{i}_embs.pt')
                    
            files = os.listdir(f'{path}embeddings/')
        else:
            print('Embeddings already exist..')
        print('Save final matrix')
        matrix = torch.empty(size=(0, 2048))
        
        files = sorted([(f"{path}embeddings/{f}", int(f.split('_')[0].strip())) \
                        for f in files],
                      key=lambda x: x[1])
        
        for i, (f, idx) in enumerate(tqdm(files)):
            matrix = torch.cat([matrix, torch.load(f)])
        
        torch.save(matrix, f'{path}data/matrix_embs.pt')

class aPY(Data):
    
    def __init__(self, name, folder, image_file, attrs_file,
                 split_file, unseen_file,
                 continuous=True, 
                 bound=None, lb=None, ub=None,
                 preprocess=False, root=None):
        super().__init__(name, folder)
        
        self._image_indices(image_file)
        self._class_indices(split_file)
        self._attribute_indices(attrs_file)
        self._class_attribute_matrix(split_file, continuous, bound, lb, ub)
        self._create_splits(split_file)
        
        if preprocess:
            self._crop_images(root)
            
    def get_seen_unseen_classes(self, unseen_file):
#         unseen_classes = []
#         with open(f'{self.folder}{self.name}splits/{unseen_file}') as f:
#             for l in f:
#                 unseen_classes += [l.strip()]
        unseen_classes = [self.idx_class[c] for c in np.unique([self.class_idx[self.imageid_class[i]] \
                                                                for i in self.unseen_samples])]
                
        seen_classes = self._get_classes(self.seen_samples)
        print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
        return seen_classes, unseen_classes
            
#     def get_seen_unseen_classes(self):
#         seen_classes = self._get_classes(self.seen_samples)
#         unseen_classes = self._get_classes(self.unseen_samples)
#         print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
#         return seen_classes, unseen_classes
        
    def _image_indices(self, image_file):
        """ Defines list of images, the dictionary (image, idx),  (idx, image), and the 
        dictionary (image id, class name).
        Refer to the notebook datasets to closely look at the variables. 
        """
    
        df = pd.read_csv(f'{self.folder}{self.name}images/{image_file}')
        df['new_names'] = df.apply(lambda x: self._join_cols(x['image_path'], 
                                                x['xmin'],
                                                x['ymin'], 
                                                x['xmax'], 
                                                x['ymax'],
                                                rounding=True), axis=1)

        self.images = list(df['new_names'])
        self.image_idx = {img:idx \
                          for idx, img in enumerate(self.images)}
        self.idx_image = {idx:img \
                          for img, idx in self.image_idx.items()}
        self.imageid_class = {row[0]:row[1]['label'] \
                              for row in df.iterrows()}
        
    def _class_indices(self, split_file): 
        """ Defines list of classes, the dictionary (class name, idx),  (idx, class name), 
        and the dictionary (class id, list imgs ids).
        Refer to the notebook datasets to closely look at the variables. 
        """

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.classes = [i[0][0] for i in splits['allclasses_names']]
        self.class_idx = {cls:idx \
                          for idx, cls in enumerate(self.classes)}
        self.idx_class = {idx:cls \
                          for cls, idx in self.class_idx.items()}

        # Match classes to list of images
        self.classid_imageid = defaultdict(list)
        for img, cls in self.imageid_class.items():
            self.classid_imageid[self.class_idx[cls]] += [img]

    def _attribute_indices(self, attrs_file): 
        """ Defines list of attributes, the dictionary (class name, idx),  (idx, class name),
        and the matrix (image id, attributes).
        Refer to the notebook datasets to closely look at the variables. 
        """
        
        self.attributes = []
        with open(f'{self.folder}{self.name}attributes/{attrs_file}') as f:
            for attr in f:
                self.attributes += [attr.strip()]
        
        self.attr_idx = {attr:idx \
                         for idx, attr in enumerate(self.attributes)}
        self.idx_attr = {idx:attr \
                         for attr, idx in self.attr_idx.items()}
        
        # CLEAN: Wrap the next three blocks into a function
        dict_imgs_attr = {}
        folder_pascal = 'VOCdevkit/VOC2008/JPEGImages/'
        with open(f'{self.folder}{self.name}attributes/apascal_train.txt') as f:
            for attr in f:
                line = attr.strip().split(' ')
                img = line[0]
                x1 = line[2]
                x2 = line[3]
                y1 = line[4]
                y2 = line[5]
                imgs_name = folder_pascal + '-'.join([img, x1, x2, y1, y2])

                dict_imgs_attr[self.image_idx[imgs_name]] = line[6:]

        with open(f'{self.folder}{self.name}attributes/apascal_test.txt') as f:
            for attr in f:
                line = attr.strip().split(' ')
                img = line[0]
                x1 = line[2]
                x2 = line[3]
                y1 = line[4]
                y2 = line[5]
                imgs_name = folder_pascal + '-'.join([img, x1, x2, y1, y2])

                dict_imgs_attr[self.image_idx[imgs_name]] = line[6:]
                
        folder_yahoo = 'yahoo_test_images/'
        with open(f'{self.folder}{self.name}attributes/ayahoo_test.txt') as f:
            for attr in f:
                line = attr.strip().split(' ')
                img = line[0]
                x1 = line[2]
                x2 = line[3]
                y1 = line[4]
                y2 = line[5]
                imgs_name = folder_yahoo + '-'.join([img, x1, x2, y1, y2])

                dict_imgs_attr[self.image_idx[imgs_name]] = line[6:]
                
        self.image_attr = np.zeros((len(dict_imgs_attr), len(self.attr_idx)))
        for i, attrs in dict_imgs_attr.items():
            self.image_attr[i] = np.array(attrs, dtype=int)
        
    
    def _create_splits(self, split_file):

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.unseen_samples = [i[0]-1 \
                          for i in splits['test_unseen_loc']]
        self.seen_samples = [i[0]-1 \
                        for i in splits['trainval_loc']]
        self.val_samples = [i[0]-1 \
                        for i in splits['val_loc']]

        self.unseen_old_to_new = {old:new \
                                  for new, old in enumerate(sorted(self.unseen_samples))}
        self.unseen_new_to_old = {new:old \
                                  for old, new in self.unseen_old_to_new.items()}

        self.seen_old_to_new = {old:new \
                                for new, old in enumerate(sorted(self.seen_samples))}
        self.seen_new_to_old = {new:old \
                                for old, new in self.seen_old_to_new.items()}
    
    def _join_cols(self, img, x1, x2, y1, y2, rounding=False):
        if rounding:
            x1 = self.normal_round(x1)
            if x1 == 0:
                x1 = 1
            x2 = self.normal_round(x2)
            if x2 == 0:
                x2 = 1
            y1 = self.normal_round(y1)
            if y1 == 0:
                y1 = 1
            y2 = self.normal_round(y2)
            if y2 == 0:
                y2 = 1
        return '-'.join([img, str(x1), str(x2),
                        str(y1), str(y2)])
    
    @staticmethod
    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    
    def _crop_images(self, root):
        print('Executing the cropping..')
        for idx, i in enumerate(tqdm(self.images)):
            path = i.split('-')
            img_path = path[0]
            if (img_path == "yahoo_test_images/bag_227.jpg"
                or img_path == "yahoo_test_images/mug_308.jpg"):
                im = Image.open(f'{root}{img_path}')
                im.save(f'{root}{i}', "JPEG")
            else:
                im = Image.open(f'{root}{img_path}')#.convert('RGB')
                img = self._crop_img(im, path)
                img.save(f'{root}{i}', "JPEG")

    @staticmethod
    def _crop_img(im, path):
        w, h = im.size
        xmin = max(float(path[1]), 0)
        ymin = max(float(path[2]), 0)
        xmax = min(float(path[3]), w)
        ymax = min(float(path[4]), h)
        
        return im.crop((xmin, ymin, xmax, ymax))
    

    
    def _class_attribute_matrix(self, split_file, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50, agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        
        #self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
        if continuous:
            print(f"Simply average on continuous attributes, no bounds.")
#             for c in self.idx_class:
#                 self.class_attributes[c] = np.mean(self.image_attr[self.classid_imageid[c]], 
#                                           axis=0)
            self.class_attributes = splits['original_att'].T    
        else:
            self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
            if bound:
                print(f"Average on binary attributes, based on bound.")
                for c in self.idx_class:
                    class_img_attr = self.image_attr[self.classid_imageid[c]]
                    class_img_attr[class_img_attr >= bound] = 1
                    class_img_attr[class_img_attr < bound] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
            elif (lb == None) or (ub == None):
                print(f"Average on binary attributes, no bound.")
                for c in self.idx_class:
                    m_round = np.rint(self.image_attr[self.classid_imageid[c]])
                    m_round[m_round == 0] = -1
                    self.class_attributes[c] = np.mean(m_round, axis=0)
            else:
                print(f"Average on binary attributes, based on lb and ub.")
                for c in self.idx_class:
                    class_img_attr = image_attr[classid_imageid[c]]
                    class_img_attr[class_img_attr >= ub] = 1
                    class_img_attr[class_img_attr <= lb] = -1
                    class_img_attr[(lb < class_img_attr) & (class_img_attr < ub)] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)                
    
class LAD(Data):
    
    def __init__(self, name, folder, image_file, labels_file, 
                 attrs_file, matrix_file, split_file, split_idx=0, 
                 continuous=True, bound=None, lb=None, ub=None,
                preprocess=False, root=None):
        super().__init__(name, folder)
        
        self._image_indices(image_file, labels_file)
        self._class_indices()
        self._attribute_indices(attrs_file, matrix_file)
        self._class_attribute_matrix(continuous, bound, lb, ub)
        self._create_splits(split_file, split_idx)
        
    def get_seen_unseen_classes(self):
        print(f"Number of seen classes: {len(self.seen_classes)}\nNumber of unseen classes: {len(self.unseen_classes)}")
        return self.seen_classes, self.unseen_classes#, self.idx_seen, self.idx_unseen
    
    def _image_indices(self, image_file, labels_file):
        """ Defines list of images, the dictionary (image, idx),  (idx, image), and the 
        dictionary (image id, class name).
        Refer to the notebook datasets to closely look at the variables. 
        """
    
        df = pd.read_csv(f'{self.folder}{self.name}attributes/{image_file}', 
                         header=None, sep=' ')
        df[1] = df[1].apply(lambda x: x[:-1])
        self.images = list(df[6])
        self.image_idx = {img:idx \
                          for idx, img in enumerate(self.images)}
        self.idx_image = {idx:img \
                          for img, idx in self.image_idx.items()}
        
        df_label = pd.read_csv(f'{self.folder}{self.name}attributes/{labels_file}', 
                               header=None, sep=',') 
        df_join = pd.merge(df, df_label, 
                           left_on=[1], right_on=[0])
        df_join['class'] = df_join[['0_y', '1_y']].agg('/'.join, axis=1)
        
        # Dictionary (image id, class name)
        image_class = {row[1][6]:row[1]['class'] \
                      for row in df_join.iterrows()}
        self.imageid_class = {self.image_idx[img]:cls \
                              for img,cls in image_class.items()}

    def _attribute_indices(self, attrs_file, matrix_file): 
        """ Defines list of attributes, the dictionary (class name, idx),  (idx, class name),
        and the matrix (image id, attributes).
        Refer to the notebook datasets to closely look at the variables. 
        """
                
        self.attributes = []
        with open(f'{self.folder}{self.name}attributes/{attrs_file}') as f:
            for attr in f:
                self.attributes += [' '.join(attr.strip()\
                                             .split(',')[:-1])]
        
        self.attr_idx = {attr:idx \
                         for idx, attr in enumerate(self.attributes)}
        self.idx_attr = {idx:attr \
                         for attr, idx in self.attr_idx.items()}
        
        self.dict_sub_imgs_attr = {}
        remove = ['[', ']', '']
        with open(f'{self.folder}{self.name}attributes/{matrix_file}') as f:
            for file in f:
                split_line = file.split(',')
                img_name = split_line[1].strip()
                img_att = split_line[-1].strip()

                list_att = list(filter(self.remove_chars, img_att.split(' ')))
                self.dict_sub_imgs_attr[self.image_idx[img_name]] = np.array(list_att, 
                                                                             dtype=int)
        # get class representation, this is not elegant and redundant with 
        # another function. We don't care at this point in time.
        class_vectors = defaultdict(list)
        for img, arr in self.dict_sub_imgs_attr.items():
            cls = self.class_idx[self.imageid_class[img]]
            class_vectors[cls] += [arr]

        self.class_attr = {}
        for c, vec in class_vectors.items():
            self.class_attr[c] = np.mean(np.array(vec), axis=0)
            
        self.dict_imgs_attr = copy.deepcopy(self.dict_sub_imgs_attr)
        for img in self.idx_image:
            if img not in self.dict_sub_imgs_attr:
                self.dict_imgs_attr[img] = self.class_attr[self.class_idx[self.imageid_class[img]]]

        self.image_attr = np.zeros((len(self.idx_image), len(self.attr_idx)))
        for i, att in self.dict_imgs_attr.items():
            self.image_attr[i] = att
        
    
        
    def _class_attribute_matrix(self, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50,
                                agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        if agg:
            class_vectors = defaultdict(list)
            for img in self.dict_sub_imgs_attr:
                cls = self.class_idx[self.imageid_class[img]]
                class_vectors[cls] += [self.image_attr[img]]
            
            self.class_attr = {}
            for c, vec in class_vectors.items():
                self.class_attr[c] = np.mean(np.array(vec), axis=0)

        self.class_attributes = np.zeros((len(self.class_attr), len(self.attr_idx)))
        for c, att in self.class_attr.items():
            self.class_attributes[c] = att
                    
    def _create_splits(self, split_file, split_idx):
        # CLEAN: wrap the blocks into two functions
        splits_list = []
        with open(f'{self.folder}{self.name}/splits/{split_file}') as f:
            for l in f:
                splits_list.append(l.strip()\
                                   .split(':')[-1]\
                                   .strip())
        
        unseen_classes = set([i.strip() \
                  for i in splits_list[split_idx].split(',')])

        idx_unseen = set()
        for c in unseen_classes:
            for cls in self.class_idx:
                if cls.startswith(c):
                    idx_unseen.add(self.class_idx[cls])
        self.unseen_classes = set([self.idx_class[c] for c in idx_unseen])

        self.unseen_samples = []
        for c in idx_unseen:
            self.unseen_samples += self.classid_imageid[c]
        
        self.unseen_old_to_new = {old:new \
                          for new, old in enumerate(sorted(self.unseen_samples))}
        self.unseen_new_to_old = {new:old \
                                  for old, new in self.unseen_old_to_new.items()}

        idx_seen = set(list(self.class_idx.values())).difference(idx_unseen)
        self.seen_classes = set([self.idx_class[c] for c in idx_seen])
        
        self.seen_samples = []
        for c in idx_seen:
            self.seen_samples += self.classid_imageid[c]
            
        self.seen_old_to_new = {old:new \
                        for new, old in enumerate(sorted(self.seen_samples))}
        self.seen_new_to_old = {new:old \
                                for old, new in self.seen_old_to_new.items()}
    
    @staticmethod
    def remove_chars(x):
        chars = ['[', ']', '']
        if (x in chars):
            return False
        else:
            return True
        
class SUN(Data):
    
    def __init__(self, name, folder, image_file, 
                 attrs_file, matrix_file,
                 split_file, unseen_file,
                 continuous=True, 
                 bound=None, lb=None, ub=None):
        super().__init__(name, folder)
        
        self._image_indices(image_file)
        self._class_indices(split_file)
        self._attribute_indices(attrs_file, matrix_file)
        self._class_attribute_matrix(split_file, continuous, bound, lb, ub)
        self._create_splits(split_file)
    
    def get_seen_unseen_classes(self, unseen_file):
#         unseen_classes = []
#         with open(f'{self.folder}{self.name}splits/{unseen_file}') as f:
#             for l in f:
#                 unseen_classes += [l.strip()]
        unseen_classes = [self.idx_class[c] for c in np.unique([self.class_idx[self.imageid_class[i]] \
                                                                for i in self.unseen_samples])]
        seen_classes = self._get_classes(self.seen_samples)
        print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
        return seen_classes, unseen_classes
        
    def _image_indices(self, image_file):
        """ Defines list of images, the dictionary (image, idx),  (idx, image), and the 
        dictionary (image id, class name).
        Refer to the notebook datasets to closely look at the variables. 
        """
        # @TODO encode the path to the images list
        self.images = scipy.io.loadmat(f'{self.folder}{self.name}attributes/{image_file}')['images']
        self.images = [i[0][0] for i in self.images]

        self.image_idx = {img:idx \
                          for idx, img in enumerate(self.images)}
        self.idx_image = {idx:img \
                          for img, idx in self.image_idx.items()}
        
        # Dictionary (image id, class name)
        self.imageid_class = {self.image_idx[img]:'_'.join(img.split('/')[1:-1]) \
                              for img in self.image_idx}
        
    def _class_indices(self, split_file): 
        """ Defines list of classes, the dictionary (class name, idx),  (idx, class name), 
        and the dictionary (class id, list imgs ids).
        Refer to the notebook datasets to closely look at the variables. 
        """

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.classes = [i[0][0] for i in splits['allclasses_names']]
        self.class_idx = {cls:idx \
                          for idx, cls in enumerate(self.classes)}
        self.idx_class = {idx:cls \
                          for cls, idx in self.class_idx.items()}

        # Match classes to list of images
        self.classid_imageid = defaultdict(list)
        for img, cls in self.imageid_class.items():
            self.classid_imageid[self.class_idx[cls]] += [img]
    
    def _attribute_indices(self, attrs_file, matrix_file): 
        """ Defines list of attributes, the dictionary (class name, idx),  (idx, class name),
        and the matrix (image id, attributes).
        Refer to the notebook datasets to closely look at the variables. 
        """
        
        self.attributes = scipy.io.loadmat(f'{self.folder}{self.name}attributes/{attrs_file}')['attributes']
        self.attr_idx = {attr[0][0]:idx \
                         for idx, attr in enumerate(self.attributes)}
        self.idx_attr = {idx:attr \
                         for attr, idx in self.attr_idx.items()}
        
        self.image_attr = scipy.io.loadmat(f'{self.folder}{self.name}attributes/{matrix_file}')['labels_cv']
        
        
    def _create_splits(self, split_file):

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.unseen_samples = [i[0]-1 \
                          for i in splits['test_unseen_loc']]
        self.seen_samples = [i[0]-1 \
                        for i in splits['trainval_loc']]

        self.unseen_old_to_new = {old:new \
                                  for new, old in enumerate(sorted(self.unseen_samples))}
        self.unseen_new_to_old = {new:old \
                                  for old, new in self.unseen_old_to_new.items()}

        self.seen_old_to_new = {old:new \
                                for new, old in enumerate(sorted(self.seen_samples))}
        self.seen_new_to_old = {new:old \
                                for old, new in self.seen_old_to_new.items()}
        
    def _class_attribute_matrix(self, split_file, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50, agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        #splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        
        self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
        if continuous:
            print(f"Simply average on continuous attributes, no bounds.")
            for c in self.idx_class:
                self.class_attributes[c] = np.mean(self.image_attr[self.classid_imageid[c]], 
                                          axis=0)
            #self.class_attributes = splits['original_att'].T    
        else:
            self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
            if bound:
                print(f"Average on binary attributes, based on bound.")
                for c in self.idx_class:
                    class_img_attr = self.image_attr[self.classid_imageid[c]]
                    class_img_attr[class_img_attr >= bound] = 1
                    class_img_attr[class_img_attr < bound] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
            elif (lb == None) or (ub == None):
                print(f"Average on binary attributes, no bound.")
                for c in self.idx_class:
                    m_round = np.rint(self.image_attr[self.classid_imageid[c]])
                    m_round[m_round == 0] = -1
                    self.class_attributes[c] = np.mean(m_round, axis=0)
            else:
                print(f"Average on binary attributes, based on lb and ub.")
                for c in self.idx_class:
                    class_img_attr = image_attr[classid_imageid[c]]
                    class_img_attr[class_img_attr >= ub] = 1
                    class_img_attr[class_img_attr <= lb] = -1
                    class_img_attr[(lb < class_img_attr) & (class_img_attr < ub)] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
        

        
class CUB(Data):
    
    def __init__(self, name, folder, 
                 image_file, 
                 attrs_file, matrix_file,
                 split_file, unseen_file,
                 continuous=True, bound=None,
                 lb=None, ub=None):
        super().__init__(name, folder)
        
        self._image_indices(image_file)
        self._class_indices(split_file)
        self._attribute_indices(attrs_file, matrix_file)
        self._class_attribute_matrix(split_file, continuous, bound, lb, ub)
        self._create_splits(split_file)
        
    def get_seen_unseen_classes(self, unseen_file):
#         unseen_classes = []
#         with open(f'{self.folder}{self.name}splits/{unseen_file}') as f:
#             for l in f:
#                 unseen_classes += [l.strip()]
        unseen_classes = [self.idx_class[c] for c in np.unique([self.class_idx[self.imageid_class[i]] \
                                                                for i in self.unseen_samples])]
        seen_classes = self._get_classes(self.seen_samples)
        print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
        return seen_classes, unseen_classes
        
    def _image_indices(self, image_file):
        """ Defines list of images, the dictionary (image, idx),  (idx, image), and the 
        dictionary (image id, class name).
        Refer to the notebook datasets to closely look at the variables. 
        """
        # @TODO encode the path to the images list
        self.images = []
        img_id = {}
        with open(f'{self.folder}{self.name}attributes/{image_file}') as f:
            for l in f:
                line = l.strip().split(' ')
                idx = line[0]
                img = line[1]
                img_id[img] = int(idx)
                self.images += [img]


        self.image_idx = {img:idx \
                          for idx, img in enumerate(self.images)}
        self.idx_image = {idx:img \
                          for img, idx in self.image_idx.items()}
        
        self.orig_new = {img_id[img]:idx \
                         for img, idx in self.image_idx.items()}
        
        # Dictionary (image id, class name)
        self.imageid_class = {self.image_idx[img]:'/'.join(img.split('/')[:-1]) \
                              for img in self.image_idx}
    
    
    def _class_indices(self, split_file): 
        """ Defines list of classes, the dictionary (class name, idx),  (idx, class name), 
        and the dictionary (class id, list imgs ids).
        Refer to the notebook datasets to closely look at the variables. 
        """

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.classes = [i[0][0] for i in splits['allclasses_names']]
        self.class_idx = {cls:idx \
                          for idx, cls in enumerate(self.classes)}
        self.idx_class = {idx:cls \
                          for cls, idx in self.class_idx.items()}

        # Match classes to list of images
        self.classid_imageid = defaultdict(list)
        for img, cls in self.imageid_class.items():
            self.classid_imageid[self.class_idx[cls]] += [img]
    
    def _attribute_indices(self, attrs_file, matrix_file): 
        """ Defines list of attributes, the dictionary (class name, idx),  (idx, class name),
        and the matrix (image id, attributes).
        Refer to the notebook datasets to closely look at the variables. 
        """
        
        self.attr_idx = {}
        att_id = {}
        with open(f'{self.folder}{self.name}attributes/{attrs_file}') as f:
            for l in f:
                line = l.strip().split(' ')
                idx = line[0]
                attr = ' '.join(line[1:])
                self.attr_idx[attr] = int(idx) - 1
                att_id[attr] = int(idx)
        
        self.attributes = list(self.attr_idx.keys())
        self.idx_attr = {idx:attr \
                         for attr, idx in self.attr_idx.items()}
        
        att_orig_new = {att_id[attr]:idx \
                        for attr, idx in self.attr_idx.items()}
        
        df = pd.read_csv(f'{self.folder}{self.name}attributes/{matrix_file}',
                         sep=' ', header=None,
                         names=['img_id', 'attr_id', 'present', 
                                'certainty', 'worker_id'])
        df['img_id'] = df['img_id'].apply(lambda x: self.orig_new[x])
        df['attr_id'] = df['attr_id'].apply(lambda x: att_orig_new[x])
        df_group = df.groupby(['img_id', 'attr_id']).agg({'present':np.mean,
                                                          'certainty':np.mean})
        
        self.image_attr = np.zeros((len(self.images), len(self.attributes)))
        att = df_group['present']
        for (img, attr), a in zip(df_group.index, att):
            self.image_attr[img, attr] = a
        
    def _create_splits(self, split_file):

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.unseen_samples = [i[0]-1 \
                          for i in splits['test_unseen_loc']]
        self.seen_samples = [i[0]-1 \
                        for i in splits['trainval_loc']]

        self.unseen_old_to_new = {old:new \
                                  for new, old in enumerate(sorted(self.unseen_samples))}
        self.unseen_new_to_old = {new:old \
                                  for old, new in self.unseen_old_to_new.items()}

        self.seen_old_to_new = {old:new \
                                for new, old in enumerate(sorted(self.seen_samples))}
        self.seen_new_to_old = {new:old \
                                for old, new in self.seen_old_to_new.items()}
    
    def _class_attribute_matrix(self, split_file, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50, agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        #splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        
        self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
        if continuous:
            print(f"Simply average on continuous attributes, no bounds.")
            for c in self.idx_class:
                self.class_attributes[c] = np.mean(self.image_attr[self.classid_imageid[c]], 
                                          axis=0)
            #self.class_attributes = splits['original_att'].T    
        else:
            self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
            if bound:
                print(f"Average on binary attributes, based on bound.")
                for c in self.idx_class:
                    class_img_attr = self.image_attr[self.classid_imageid[c]]
                    class_img_attr[class_img_attr >= bound] = 1
                    class_img_attr[class_img_attr < bound] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
            elif (lb == None) or (ub == None):
                print(f"Average on binary attributes, no bound.")
                for c in self.idx_class:
                    m_round = np.rint(self.image_attr[self.classid_imageid[c]])
                    m_round[m_round == 0] = -1
                    self.class_attributes[c] = np.mean(m_round, axis=0)
            else:
                print(f"Average on binary attributes, based on lb and ub.")
                for c in self.idx_class:
                    class_img_attr = image_attr[classid_imageid[c]]
                    class_img_attr[class_img_attr >= ub] = 1
                    class_img_attr[class_img_attr <= lb] = -1
                    class_img_attr[(lb < class_img_attr) & (class_img_attr < ub)] = 0

                    self.class_attributes[c] = np.mean(class_img_attr, 
                                                  axis=0)
        
        
class AwA(Data):
    
    def __init__(self, name, folder,
                 image_file,
                 attrs_file, matrix_file,
                 split_file, unseen_file,
                 continuous=True, 
                 bound=None, lb=None, ub=None):
        super().__init__(name, folder)
        
        self._image_indices(image_file)
        self._class_indices(split_file)
        self._attribute_indices(attrs_file, matrix_file)
        self._class_attribute_matrix(split_file, continuous, bound, lb, ub)
        self._create_splits(split_file)
        
    def get_seen_unseen_classes(self, unseen_file):
#         unseen_classes = []
#         with open(f'{self.folder}{self.name}splits/{unseen_file}') as f:
#             for l in f:
#                 unseen_classes += [l.strip()]
        unseen_classes = [self.idx_class[c] for c in np.unique([self.class_idx[self.imageid_class[i]] \
                                                                for i in self.unseen_samples])]
                
        seen_classes = self._get_classes(self.seen_samples)
        print(f"Number of seen classes: {len(seen_classes)}\nNumber of unseen classes: {len(unseen_classes)}")
        return seen_classes, unseen_classes
        
    def get_class_attribute_matrix(self, split_file, continuous=True, bound=None, 
                                   lb=None, ub=None):
        if continuous:
            asterisks = np.where(self.class_attributes == -1)
            self.class_attributes[asterisks] = 0
            print(f"Continuous - Number of unknown: {len(asterisks[0])}")
            return self.class_attributes
        else:
            matrix = copy.deepcopy(self.class_attributes)
            asterisks = np.where(matrix == -1)
            print(f"Number of unknown: {len(asterisks[0])}")
            
            if bound:
                matrix[matrix >= bound] = 1
                matrix[matrix < bound] = -1
                
                return matrix
            
            elif (lb == None) or (ub == None):
                matrix = np.rint(matrix)
                matrix[matrix == 0] = -1
                matrix[asterisks] = 0
                return matrix
            
            else:
                less_lb = np.where(matrix <= lb)
                more_ub = np.where(matrix >= ub)
                btw_bounds = np.where((lb < matrix) & (matrix < ub))
                
                
                matrix[less_lb] = -1
                matrix[more_ub] = 1
                matrix[btw_bounds] = 0
                matrix[asterisks] = 0
                
                return matrix
        
    def _image_indices(self, image_file):
        """ Defines list of images, the dictionary (image, idx),  (idx, image), and the 
        dictionary (image id, class name).
        Refer to the notebook datasets to closely look at the variables. 
        """
        # @TODO encode the path to the images list
        self.classes = []
        with open(f'{self.folder}{self.name}attributes/{image_file}') as f:
            for l in f:
                self.classes += [l.strip().split('\t')[-1]]
        
        self.images = []
        img_cls = {}
        for c in sorted(self.classes):
            imgs = os.listdir(f'{self.folder}{self.name}images/{c}/')
            self.images += [f'images/{c}/{im}' for im in imgs]
            for i in imgs:
                img_cls[f'images/{c}/{i}'] = c


        self.image_idx = {img:idx \
                          for idx, img in enumerate(self.images)}
        self.idx_image = {idx:img \
                          for img, idx in self.image_idx.items()}
        
        # Dictionary (image id, class name)
        self.imageid_class = {self.image_idx[img]:img_cls[img] \
                              for img in self.image_idx}
    
    def _class_indices(self, split_file): 
        """ Defines list of classes, the dictionary (class name, idx),  (idx, class name), 
        and the dictionary (class id, list imgs ids).
        Refer to the notebook datasets to closely look at the variables. 
        """

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.classes = [i[0][0] for i in splits['allclasses_names']]
        self.class_idx = {cls:idx \
                          for idx, cls in enumerate(self.classes)}
        self.idx_class = {idx:cls \
                          for cls, idx in self.class_idx.items()}

        # Match classes to list of images
        self.classid_imageid = defaultdict(list)
        for img, cls in self.imageid_class.items():
            self.classid_imageid[self.class_idx[cls]] += [img]
    
    def _attribute_indices(self, attrs_file, matrix_file): 
        """ Defines list of attributes, the dictionary (class name, idx),  (idx, class name),
        and the matrix (image id, attributes).
        Refer to the notebook datasets to closely look at the variables. 
        """
        
        self.attr_idx = {}
        with open(f'{self.folder}{self.name}attributes/{attrs_file}') as f:
            for idx, l in enumerate(f):
                attr = l.strip().split('\t')[-1]
                self.attr_idx[attr] = idx

        self.attributes = list(self.attr_idx.keys())
        self.idx_attr = {idx:attr \
                         for attr, idx in self.attr_idx.items()}
        
        self.class_attributes = np.zeros((len(self.classes), len(self.attr_idx)))
        with open(f'{self.folder}{self.name}attributes/{matrix_file}') as f:
            for i,l in enumerate(f):
                if matrix_file == 'predicate-matrix-continuous.txt':
                    self.class_attributes[i] = np.array(l.strip().split('  '),
                                               dtype=float)
                else:
                    self.class_attributes[i] = np.array(l.strip().split(' '), dtype=int)
                
        self.image_attr = np.zeros((len(self.images), len(self.attributes)))
        for i in self.idx_image:
            self.image_attr[i] = self.class_attributes[self.class_idx[self.imageid_class[i]]]
            
        
    def _create_splits(self, split_file):

        splits = scipy.io.loadmat(f'{self.folder}{self.name}splits/{split_file}')
        self.unseen_samples = [i[0]-1 \
                          for i in splits['test_unseen_loc']]
        self.seen_samples = [i[0]-1 \
                        for i in splits['trainval_loc']]

        self.unseen_old_to_new = {old:new \
                                  for new, old in enumerate(sorted(self.unseen_samples))}
        self.unseen_new_to_old = {new:old \
                                  for old, new in self.unseen_old_to_new.items()}

        self.seen_old_to_new = {old:new \
                                for new, old in enumerate(sorted(self.seen_samples))}
        self.seen_new_to_old = {new:old \
                                for old, new in self.seen_old_to_new.items()}
        
        
    def _class_attribute_matrix(self, continuous=True, bound=None,
                                lb=None, ub=None, n_examples=50, agg=False):
        """Builds the class-feature matrix by averaging the image, attribute
        representation wrt the class
        
        :param continuous: if False attributes are rounded according to bound or
        lb and ub.
        :param bound: threshold for binary attributes
        :param lb: threshold for negative attributes
        :param ub: threshold for positive attributes
        :param n_examples: number of example per class, not in use yes. Add if needed
        """
        
        return self.class_attributes