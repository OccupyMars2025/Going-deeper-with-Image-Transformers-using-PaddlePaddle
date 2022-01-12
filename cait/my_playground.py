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



import paddle
import paddle.nn as nn

proj = nn.Conv2D(3, 200, kernel_size=[16, 16], stride=[16, 16])
x = paddle.randn([4, 3, 224+7, 224+3])
y = proj(x)
print(y.shape)
