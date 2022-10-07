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
# Caution: when reproducing the model, if you want to pass all forward and backward checks, you need to remove all random factors (pay attention to the arguments of cait_models, set some arguments to 0 to remove the random factors)

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


# 2. Now, the conda environment is configured and you can evaluate cait-XXS24-224 model on [imagenet validation dataset](https://aistudio.baidu.com/aistudio/datasetdetail/68594). The "resume" and "model" arguments are copied from the source code as the following shows:                 
    
<img width="960" alt="image" src="https://user-images.githubusercontent.com/31559413/194271488-3f2745eb-eb92-4c4c-a9d9-43ed24fa9376.png">
You need to use your own `class ImageNetDataset` in the function `build_dataset`
<img width="960" alt="image" src="https://user-images.githubusercontent.com/31559413/194288641-fa1d8c1c-bd1f-490b-a74c-cc70c8450258.png">

## run the the following command in the CaiT_torch directory on my local machine (windows 11) to evaluate  cait-XXS24-224 model on imagenet validation dataset.

```shell
python main.py --eval --pretrained  --device  cpu     --model  cait_XXS24_224  --data-path  C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\imagenet_dataset\ILSVRC2012_img_val   --train-info-txt  C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\imagenet_dataset\train_list_empty.txt     --val-info-txt  C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\imagenet_dataset\val_list.txt
```
    
## If you want to set "--model"  to "cait_XXS24_224", you need to "import cait_models" in main.py. Just like if you want to use the default value of "--model", then you need to "import models" in main.py, thus the decorator "@register_model" can work.

## The evaluation results are as follows: 
```shell
Not using distributed mode
Namespace(ThreeAugment=False, aa='rand-m9-mstd0.5-inc1', attn_only=False, batch_size=64, bce_loss=False, clip_grad=None, color_jitter=0.3, cooldown_epochs=10, cutmix=1.0, cutmix_minmax=None, data_path='C:\\Users\\Administrator\\Desktop\\contests\\20220715_paddle_lwfx_7th\\imagenet_dataset\\ILSVRC2012_img_val', data_set='IMNET', decay_epochs=30, decay_rate=0.1, device='cpu', dist_eval=False, dist_url='env://', distillation_alpha=0.5, distillation_tau=1.0, distillation_type='none', distributed=False, drop=0.0, drop_path=0.1, epochs=300, eval=True, eval_crop_ratio=0.875, finetune='', inat_category='name', input_size=224, lr=0.0005, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, min_lr=1e-05, mixup=0.8, mixup_mode='batch', mixup_prob=1.0, mixup_switch_prob=0.5, model='cait_XXS24_224', model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False, momentum=0.9, num_workers=10, opt='adamw', opt_betas=None, opt_eps=1e-08, output_dir='', patience_epochs=10, pin_mem=True, pretrained=True, recount=1, remode='pixel', repeated_aug=True, reprob=0.25, resplit=False, resume='', sched='cosine', seed=0, smoothing=0.1, src=False, start_epoch=0, teacher_model='regnety_160', teacher_path='', train_info_txt='C:\\Users\\Administrator\\Desktop\\contests\\20220715_paddle_lwfx_7th\\imagenet_dataset\\train_list_empty.txt', train_interpolation='bicubic', train_mode=True, unscale_lr=False, val_info_txt='C:\\Users\\Administrator\\Desktop\\contests\\20220715_paddle_lwfx_7th\\imagenet_dataset\\val_list.txt', warmup_epochs=5, warmup_lr=1e-06, weight_decay=0.05, world_size=1)
C:\Users\Administrator\Anaconda3\envs\paddle-2.3-pytorch-1.8-python-3.7-env\lib\site-packages\torchvision\transforms\transforms.py:258: UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
  "Argument interpolation should be of type InterpolationMode instead of int. "
C:\Users\Administrator\Anaconda3\envs\paddle-2.3-pytorch-1.8-python-3.7-env\lib\site-packages\torch\utils\data\dataloader.py:477: UserWarning: This DataLoader will create 10 worker processes in total. Our suggested max number of worker in current system is 8 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Creating model: cait_XXS24_224
number of params: 11956264
C:\Users\Administrator\Anaconda3\envs\paddle-2.3-pytorch-1.8-python-3.7-env\lib\site-packages\torch\cuda\amp\grad_scaler.py:116: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
  warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
C:\Users\Administrator\Anaconda3\envs\paddle-2.3-pytorch-1.8-python-3.7-env\lib\site-packages\torch\cuda\amp\autocast_mode.py:118: UserWarning: torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.
  warnings.warn("torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
Test:  [  0/521]  eta: 4:52:11  loss: 0.7475 (0.7475)  acc1: 79.1667 (79.1667)  acc5: 94.7917 (94.7917)  time: 33.6491  data: 11.6625
Test:  [ 10/521]  eta: 3:03:42  loss: 0.8712 (0.8758)  acc1: 78.1250 (78.5985)  acc5: 93.7500 (93.8447)  time: 21.5711  data: 1.0618
Test:  [ 20/521]  eta: 2:59:58  loss: 0.8682 (0.8856)  acc1: 78.1250 (78.2242)  acc5: 93.7500 (93.7004)  time: 20.9494  data: 0.0009
Test:  [ 30/521]  eta: 2:57:02  loss: 0.8466 (0.8830)  acc1: 79.1667 (78.4946)  acc5: 94.7917 (94.1196)  time: 21.6704  data: 0.0000
Test:  [ 40/521]  eta: 2:54:31  loss: 0.8948 (0.8862)  acc1: 78.1250 (78.2774)  acc5: 94.7917 (94.1819)  time: 21.9972  data: 0.0000
Test:  [ 50/521]  eta: 2:52:31  loss: 0.9192 (0.8901)  acc1: 78.1250 (78.2475)  acc5: 93.7500 (94.0359)  time: 22.5098  data: 0.0008
```

## Caution: adding convert('RGB') is essential, or it probably will report an error if an image is not in "RGB" format
```python
image = Image.open(image_path).convert('RGB')
```
```shell
==================== before transformation <PIL.JpegImagePlugin.JpegImageFile image mode=L size=500x375 at 0x29C3C2B2438>
==================== The image that cause the error: C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\imagenet_dataset\ILSVRC2012_img_val\ILSVRC2012_val_00001769.JPEG
==================== The image that cause the error: <PIL.JpegImagePlugin.JpegImageFile image mode=L size=500x375 at 0x29C3C2B2438>
None
```
## Error: No mudule named ppimm is found 
## Solution is as follows: add the path of the folder where ppimm is located to the PYTHONPATH. ppimm is just the same as timm but written with PaddlePaddle
if any module is not found, maybe you can to C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\Going-deeper-with-Image-Transformers-using-PaddlePaddle\CaiT_paddle  and  C:\Users\Administrator\Desktop\contests\20220715_paddle_lwfx_7th\Going-deeper-with-Image-Transformers-using-PaddlePaddle\CaiT_torch to the PYTHONPATH
<img width="288" alt="image" src="https://user-images.githubusercontent.com/31559413/194469915-d2121833-db34-4091-a989-f65fd7308dd6.png">
<img width="538" alt="image" src="https://user-images.githubusercontent.com/31559413/194469997-eede3062-af37-4c5f-a7ca-14ddcdcb7bcf.png">
