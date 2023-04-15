# DQN_test  

***
##说明  

大致了解了一下强化学习中的经典DQN  
用找到的代码进行简单的尝试  
1. DQN取自莫烦python教程
2. CartPole为gym自带简单环境


***
##食用指南  

1.首先需要在conda中创建新环境  
```conda create -n {your_env_name} python={X.X}```#请将{ }中的内容进行替换  
2.激活新创建的环境  
```conda activate {your_env_name}```#请将{ }中的内容进行替换  
3.安装相关依赖包  
```conda install gym```  
```pip install gym[classic_control]```  
可能还需要numpy等包(多数情况下安装gym时已经安装了numpy)  
```conda install numpy```  
4.运行