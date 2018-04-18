## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random
import pickle

#from setup_cifar import CIFAR, CIFARModel
#from setup_mnist import MNIST, MNISTModel
from classify_image import ImageNet, InceptionModel, Imageset

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

class newdata:
    def __init__(self, od, ol, ad, al):
        self.origin_data = np.array(od)
        self.origin_label= np.squeeze(np.array(ol))
        self.adv_data = np.array(ad)
        self.adv_label= np.array(al)

#        print(self.origin_data.shape)
        print(self.origin_label.shape)
#        print(self.adv_data.shape)
        print(self.adv_label.shape)

    def add_target(self,target):
        self.target = target
        print(target)


def generate_data(model, data, samples, targeted=True, start=0, inception=False, batch_size = 10):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        seq = random.sample(range(200,1001), batch_size)
#        print('target:',seq)

        for j in seq:
            inputs.append(data.train_data[start+i])
            label = np.zeros(data.train_labels.shape[1])
            label[j] = 1
            targets.append(label)

    inputs = np.array(inputs)
    targets = np.array(targets)
    idx = np.argmax(targets, axis = 1)

    return inputs, targets

def evaluate(model, data, label):
    tt1 = time.time()
    predict = model.predict(data)
    tt2 = time.time()
    print('time to predict:',tt2-tt1)
    pl = np.argmax(predict,axis = 1) 
    l = np.argmax(label,axis = 1)
    error = 0
    num = pl.shape[0]
    for i in range(num):

        if pl[i] != l[i]:
            error +=1

    accuracy = 1 - float(error)/num
    print('accuracy:',accuracy)

    return accuracy

origin_data = []
adv_data = []
origin_label = []
adv_label = []
ut_data = []
ut_label = []

tar = []
ut_tar = []

samples = 50
start = 0
confidence = 0
bs = 9
mi = 1000

filename = '1sample50bs9.pkl'
utfile = 'ut_'+filename

if __name__ == "__main__":
    with tf.Session() as sess:

        data, model = ImageNet(), InceptionModel(sess)
        inputs, targets = generate_data(model, data, samples=samples, targeted=True,
                                        start=start, inception=True, batch_size = bs)

        tar = np.argmax(targets, axis = 1)

#        evaluate(model.model, data.test_data[0:1000], data.test_labels[0:1000])

#        new_data=newdata(data.test_data[0:1000], data.test_labels[0:1000], data.test_data[0:1000], data.test_labels[0:1000] )

        attack = CarliniL2(sess, model, batch_size=bs, max_iterations = mi, confidence=confidence)
#        attack = CarliniL0(sess, model, max_iterations = mi)
#        attack = CarliniLi(sess, model)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(int(len(adv)/bs)):
#            origin_data.append(inputs[i*bs])
            origin_label.append(model.pred(inputs[i*bs:i*bs+1],True))

#            adv_data.append(adv[i*bs:i*bs+bs])
            adv_label.append(model.pred(adv[i*bs:i*bs+bs], True))

            temp = np.tile(np.array(inputs[i]),(bs,1,1,1))
            diff = np.array(adv[i*bs:i*bs+bs]) - temp
            noise = (diff**2).sum(axis = 3).sum(axis = 2).sum(axis = 1)
            idx = np.argmin(noise)

#            print('*****************')
#            ut_data.append(adv[i*bs+idx])
            ut_label.append(model.pred(adv[i*bs+idx:i*bs+idx+1], True))
            ut_tar.append(tar[i*bs+idx])
#            print('*****************')

#        adv_data = np.reshape(adv_data,(samples*bs,299,299,3))
        adv_label = np.reshape(adv_label,(samples*bs,1008))

        ut_label = np.squeeze(np.array(ut_label))
#        print(ut_label.shape)
      
'''
        for i in range(len(adv)):
#            print("Valid:")
#            show(inputs[i])
#            print("Adversarial:")
#            show(adv[i])

#            print("Classification:", model.model.predict(adv[i:i+1]))
#            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
'''

new_data=newdata(origin_data, origin_label, adv_data, adv_label )
new_data.add_target(tar)
print('------------------')
new_ut = newdata(origin_data, origin_label, ut_data, ut_label)
new_ut.add_target(ut_tar)
#print(ut_label)

f = open(filename,'wb')
pickle.dump(new_data,f)
f.close

f = open(utfile,'wb')
pickle.dump(new_ut,f)
f.close

