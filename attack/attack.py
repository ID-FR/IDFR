try :
    from tensorflow import keras
except:
    import keras
import argparse
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod,BasicIterativeMethod,CarliniLInfMethod
from keras import backend as K
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)



parser = argparse.ArgumentParser(description="对抗样本生成脚本，"
                                             "支持attack：cw,fgsm,bim"
                                             "支持数据集：cifar10，mnist")
parser.add_argument('--dataset',help="数据集名称",choices=['cifar10','mnist','svhn'],default='cifar10',type=str)
parser.add_argument('--attack_method',help="攻击算法",choices=['fgsm','cw','bim'],default='fgsm',type=str)
parser.add_argument('--model_path',help="攻击的模型地址",type=str,required=True)
parser.add_argument('--max_iter',help="攻击迭代次数",default=10,type=int)
parser.add_argument('--eps',help="扰动比例",default=0.03,type=float)
args = parser.parse_args()


# 分配计算空间
# import os
# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)

model_path = args.model_path
attack_method = args.attack_method
max_iter = args.max_iter
dataset = args.dataset
eps = args.eps

# 加载模型
model = keras.models.load_model(model_path)
# 加载数据
if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train_mean = np.mean(x_train, axis=0)
    x_test -= x_train_mean
    x_train -= x_train_mean
elif dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
elif dataset == 'svhn':
    from scipy.io import loadmat
    test = loadmat("./test_32x32.mat")
    train = loadmat("./train_32x32.mat")
    def reformat(samples, labels):
        samples = np.transpose(samples, (3, 0, 1, 2))
        labels = keras.utils.to_categorical(labels-1, 10)
        return samples, labels
    x_train, y_train = reformat(train['X'], train['y'])
    x_test, y_test = reformat(test['X'], test['y'])
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train_mean = np.mean(x_train, axis=0)
    x_test -= x_train_mean
    x_train -= x_train_mean

# 原始样本的精度
performance = model.evaluate(x_test,y_test,verbose=2)
print("原始样本的精度:",performance[1])

# 将分类器载入
classifier = KerasClassifier(model=model, clip_values=(-1, 1))
attack = {
    'fgsm' : FastGradientMethod(classifier=classifier, eps=eps),
    'bim' : BasicIterativeMethod(classifier=classifier, eps=eps, eps_step=0.01, max_iter=max_iter),
    'cw' : CarliniLInfMethod(classifier=classifier, eps=eps, max_iter=max_iter, learning_rate=0.01)
}
# 设置攻击算法
attacking = attack[attack_method]
# 攻击x_test 生成x_test_adv
print("攻击原始样本")
data = np.append(x_train,x_test,axis=0)
x_adv = attacking.generate(data)
# 对抗样本的精度
label = np.append(y_train,y_test,axis=0)
performance = model.evaluate(x_adv,label,verbose=2)
print("对抗样本的精度:",performance[1])

# 保存对抗样本
print("保存对抗样本")

try:
    np.save("./adv_data/"+dataset+"/"+attack_method+"/"+str(eps)+'x_adv.npy',x_adv)
except:
    import os
    os.makedirs("./adv_data/"+dataset+"/"+attack_method)
    np.save("./adv_data/"+dataset+"/"+attack_method+"/"+str(eps)+'x_adv.npy',x_adv)

print("保存成功！")
print("数据集：",dataset,"攻击算法：",attack_method," 攻击后精度：",performance[1],"迭代次数:",max_iter)

