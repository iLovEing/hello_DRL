# DQN

## 算法原理
`DQN`属于value based算法，使用神经网络计算action value， 实际上是求解贝尔曼最优方程：  
1. 设估计Q value的函数为 $\hat{q} (s, a, w)$, 其中 $w$为参数，value based方法的优化目标为（这里，前一项称为target Q value，后一项为policy Q value）:  
$J(\omega ) = E[{(R + \gamma max_{a\in A(S\prime )}\hat{q}(S\prime, a, w) - \hat{q}(S, a, w))}^{2}]$  
2. 使用梯度下降法求解目标函数，由于式中含有max项，直接求梯度难度很大，因此DQN提出，**固定计算target value的网络**，只对policy Q value的网络参数求梯度，后者用梯度下降更新，前者用动量赋值更新。这样，使用随机梯度下降更新参数：
$\omega_{t+1} = \omega_{t} - \alpha [-2(r+\gamma max_{a\in A(s\prime )}\hat{q}(s\prime, a, W_{T}) - \hat{q}(S, a, w))\nabla_{\omega} \hat{q}(s, a, \omega )]$  
其中, $W_{T}$ 为固定的网络参数, $w$ 为求解网络参数，其余变量由采样数据 $(s, a, r, s\prime)$ 得到

**注意**: 这里算法所需的数据不需要由policy Q network采样，因此`DQN` 是**on-line、off-policy**的算法。

## 代码说明
1. Q network训练就是用policy Q newwork计算的Q value去逼近target Q value，可以用均方差损失，这里使用pytorch中的SmoothL1Loss:
  - target Q value: $r+\gamma max_{a\in A(s\prime )}\hat{q}(s\prime, a, W_{T})$
  - policy Q value: $\hat{q}(s, a, w)$

2. DQN中target Q value涉及到求解max q，因此不适合处理连续动作空间，这里只以 `CartPole-v1` 为案例。输入为state，输出为各个action的Q value.
   
3. 伪代码
   1. 初始化：初始化replay buffer，随机初始化policy Q network、target Q network，这里两者参数保持一致
   2. 循环采样、计算:
      1. 采样数据得到，这里策略使用policy Q network，方法使用 $\epsilon - greedy$ 或者半随机采样，总之，需要同时保证采样的探索性和有效性，需要根据不同的模型判断
      2. 采样得到一定数量的数据后，开始更新网络
      3. 计算target Q value和policy Q value，使用梯度下降更新policy Q network
      4. 使用policy Q network的参数动量更新target Q network
4. 参考代码  
   [pytorch官方DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)


## 训练结果
- CartPole-v1  
![CartPole-v1 train reward](https://github.com/iLovEing/hello_RL/blob/main/DQN/train_log/CartPole-v1_reward.png)
