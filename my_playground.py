# import paddle
# import paddle.nn as nn
#
#
# from paddle.nn.initializer import TruncatedNormal, Constant
#
# # 参数初始化配置
# trunc_normal_ = TruncatedNormal(std=.02)
# zeros_ = Constant(value=0.)
# ones_ = Constant(value=1.)
#
#
#
#
# class MyLayer(nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#
#         self.pos_embed = self.create_parameter(
#             shape=[1, 2, 3], default_initializer=trunc_normal_)
#         self.add_parameter("pos_embed", self.pos_embed)
#         self.cls_token = self.create_parameter(
#             shape=[1, 1, 4], default_initializer=trunc_normal_)
#         self.add_parameter("cls_token", self.cls_token)
#         zeros_(self.pos_embed)
#         ones_(self.cls_token)
#
#
# if __name__ == "__main__":
#     my_layer = MyLayer()
#     print(my_layer.parameters())



# import paddle
# import paddle.nn as nn
#
# proj = nn.Conv2D(3, 200, kernel_size=[16, 16], stride=[16, 16])
# x = paddle.randn([4, 3, 224+7, 224+3])
# y = proj(x)
# print(y.shape)


# import numpy as np
# from PIL import Image
# from paddle.vision.transforms import RandomResizedCrop
#
# transform = RandomResizedCrop(224)
#
# fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))
#
# fake_img = transform(fake_img)
# print(fake_img.size)

# import paddle
#
#
# num_classes = 7
# x = paddle.randint(0, num_classes, shape=[4])
# off_value, on_value = 0, 1
# out = paddle.full(shape=[x.shape[0], num_classes], fill_value=off_value)
# image_index_list = list(range(x.shape[0]))
# x_list = paddle.cast(x, dtype='int64').tolist()
# out[image_index_list, x_list] = on_value
# print(x, out, sep='\n')