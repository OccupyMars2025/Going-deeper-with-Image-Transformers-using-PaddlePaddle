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

# 2022/10/1:
lwfx, 7th       


<img width="572" alt="image" src="https://user-images.githubusercontent.com/31559413/194200939-edfec100-0272-41fb-b5bb-6de6794045a3.png">


