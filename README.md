TODO:     
1.  modify  how  to  generate random numbers (originally use torch.rand,  you have to  change it to paddle.rand, don't forget ) in  _add_noise  in  task3-Going-deeper-with-Image-Transformers\CaiT_paddle\ppimm\scheduler\cosine_lr.py    
2.  in  05_test_backward.py,  because  paddle's  lr_schedular  use step() in  its  __init__ method,  so in  torch, after creating the lr_schedular, I add lr_schedular.step() to  make these two  lr_schedulars synchronous, how to solve this problem ???


# Going-deeper-with-Image-Transformers-using-PaddlePaddle     
task 11.      
implement paper "Going deeper with Image Transformers" with PaddlePaddle       
The implemented model is called "CaiT"(Class Attention Image Transformer)      
contest website:      
https://github.com/PaddlePaddle/Paddle/issues/37401    
https://aistudio.baidu.com/aistudio/competition/detail/126/0/task-definition      
AI studio:    
https://aistudio.baidu.com/aistudio/projectdetail/3383308?contributionType=1
         
         
current training log:      
epoch: 1, batch_id: 0, loss is: [4.3869734], acc is: [0.15625]   (batch_size is 64)


2022/5/9:    
problem:    
https://github.com/PaddlePaddle/Paddle/issues/42609

# 2022/10/1:  lwfx, 7th, paper 59

# [validation dataset](https://aistudio.baidu.com/aistudio/datasetdetail/68594)


<img width="572" alt="image" src="https://user-images.githubusercontent.com/31559413/194200939-edfec100-0272-41fb-b5bb-6de6794045a3.png">


# 1. setup a conda environment on local machine (windows 11):    
(first update conda version.)    
conda update -n base -c defaults conda     
(pay attention to all kinds of versions)     
conda create --name paddle-2.3-pytorch-1.8-python-3.7-env  python==3.7    
conda activate paddle-2.3-pytorch-1.8-python-3.7-env              
(Pytorch LTS version 1.8.2 is only supported for Python <= 3.8)           
conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts              
conda install paddlepaddle==2.3.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/             
(install the required version of timm)            
pip install timm==0.3.2             


# 2. Now, the conda environment is configured and you can evaluate cait-XXS24-224 model on imagenet validation dataset. The "resume" and "model" arguments are copied from the source code as the following shows:                 
    
<img width="960" alt="image" src="https://user-images.githubusercontent.com/31559413/194271488-3f2745eb-eb92-4c4c-a9d9-43ed24fa9376.png">

```shell
python main.py --eval --resume  https://dl.fbaipublicfiles.com/deit/XXS24_224.pth  --model  cait_XXS24_224  --data-path  C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\imagenet_dataset
```
