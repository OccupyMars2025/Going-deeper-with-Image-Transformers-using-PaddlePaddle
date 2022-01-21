######  脚本任务使用注意事项  ######
# 1.输出结果的体积上限为20GB，超过上限可能导致下载输出失败，建议仅保存必要文件.
# 2.脚本任务单次任务最大运行时长为72小时（三天）.
# 3.在使用单机四卡或双击四卡时可不配置GPU编号，默认启动所有可见卡；如需配置GPU编号，单机四卡的GPU编号为0,1,2,3；双机四卡的GPU编号为0,1.
# 更多详细教程请在左侧目录的run.py、run.sh文件或在AI Studio文档(https://ai.baidu.com/ai-doc/AISTUDIO/Ik3e3g4lt)中进行查看.

# 代码案例：FashionMNIST-分布式训练(Paddle 2.1.2)
import os
import gzip
import numpy as np
from PIL import Image
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import transforms

def load_mnist(path, kind='train'):
    """加载mnist数据集"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)

    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, transform=None, kind='train'):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        # 数据集文件目录
        # datasets_prefix为数据集的根路径，完整的数据集文件路径是由根路径和相对路径拼接组成的。
        # 相对路径获取方式：请在编辑项目状态下通过点击左侧导航「数据集」中文件右侧的【复制】按钮获取.
        # datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'
        # train_datasets =  datasets_prefix+'通过路径拷贝获取真实数据集文件路径'
        datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'
        train_datasets = datasets_prefix + 'data7688/'
        self.images, self.labels = load_mnist(train_datasets, kind)
        self.transform = transform
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image, label = self.images[index], self.labels[index]
        image = np.reshape(image, [28, 28])
        image = Image.fromarray(image.astype('uint8'), mode='L')
        if self.transform is not None:
            image = self.transform(image)
        return image.astype("float32"), label.astype('int64')
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.images)
# 数据集预处理
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MyDataset(transform=transform)
batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=32, shuffle=True)
def train():
    # 设置支持多卡训练
    dist.init_parallel_env()
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    model = paddle.vision.models.LeNet()
    # 设置支持多卡训练
    model = paddle.DataParallel(model)
    # 设置优化方法
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=0.1)
    # 开始训练
    for epoch in range(10):
        for batch_id, (img, label) in enumerate(train_loader()):
            output = model(img)
            # 计算损失值
            loss = F.cross_entropy(output, label)
            loss.backward()
            if batch_id % 100 == 0:
                print("Epoch {}: batch_id {}, loss {}".format(epoch, batch_id, loss.numpy()))
            optimizer.step()
            optimizer.clear_grad()
        # 保存模型
        # 输出文件路径
        # 任务完成后平台会自动把output_dir目录所有文件压缩为tar.gz包，用户可以通过「下载输出」将输出结果下载到本地.
        # output_dir = "/root/paddlejob/workspace/output/"
        paddle.save(model.state_dict(),
                    os.path.join("/root/paddlejob/workspace/output/", "{}_model.pdparams".format(epoch)))

if __name__ == '__main__':
    train()
# 启动命令：
# 1. python 指令
# ---------------------------------------单机单卡-------------------------------------------
# python mnist.py
# ---------------------------------------单机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1,2,3" run.py
# ---------------------------------------双机四卡-------------------------------------------
# 方式一（不配置GPU编号）：python -m paddle.distributed.launch run.py
# 方式二（配置GPU编号）：python -m paddle.distributed.launch --gpus="0,1" run.py
# 2. shell 命令
# 使用run.sh或自行创建新的shell文件并在对应的文件中写下需要执行的命令(需要运行多条命令建议使用shell命令的方式)。
# 以单机四卡不配置GPU编号为例，将单机四卡方式一的指令复制在 run.sh 中，并在启动命令填写 bash run.sh 提交任务即可。