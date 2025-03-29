# REINFORCE

## 算法原理
`REINFORCE`属于policy gradient类算法， 其目标是直接训练actor网络指导agent行动，网络输入是state，输出为action或其分布：  
1. 策略梯度  
$\nabla_{\theta }J(\theta ) = E_{S\sim \eta , a\sim \pi } [\nabla_{\theta } \ln \pi (A\mid S, \theta )q_{\pi }(S, A) ]$  ----- :one:  
2. 使用梯度上升方法更新 $\theta $  
$\theta_{t+1} = \theta_{t} + \alpha E_{S\sim \eta , a\sim \pi } [\nabla_{\theta } \ln \pi (A\mid S, \theta )q_{\pi }(S, A) ]$   ----- :two:  
3. 使用随机梯度代替随机变量  
$\theta_{t+1} = \theta_{t} + \alpha [\nabla_{\theta } \ln \pi (a_{t}\mid s_{t}, \theta )q_{\pi }(a_{t}, s_{t}) ]$   ----- :three:  

在上式中, $s_{t}, a_{t}$ 由单次采样的到, $\pi$ 及其梯度由策略函数得到，关键是计算Q value.  
`REINFORCE` 的思路非常朴素，用Monte Carlo方法估计Q value，对一个episode采样至结束，即可估计出这条路径上每个 $q(s, a)$ 的值。

**注意**: 这里需要对 $\pi$ 进行采样，且基于MC的方法需要采样至episode结束，因此`REINFORCE` 是**off-line、on-policy**的算法。

## 代码说明
1. 用神经网络替代策略函数，根据随机梯度下降优化式 :three: 中的导数项反推, 写出伪损失函数:  
$loss = - \ln \pi (a_{t}\mid s_{t}, \theta )q_{\pi }(a_{t}, s_{t}) $
   
2. 伪代码
   1. 初始化：随机初始化策略网络 $\pi$
   2. 对每个episode (这里每个episode后 $\pi$ 参数即时更新，省略下标均用 $\pi$ 代表)：
      1. 根据 $\pi$ 采样至结束，记录每一步的action和prob
      2. 从后到前，计算每一步 $q(s, a)$ 的return，作为Q value
      3. 根据loss公式计算loss，并反向传播

3. tricks & discuss
   1. `CartPole-v1`是离散动作空间，网络输入为state，输出动作为每个动作的概率，训练时对模型输出概率进行采样，推理时直接取最大概率的动作。  
`InvertedPendulum-v5`是连续动作空间，网络输入为state，输出动作采样的mean和std，训练时使用高斯分布对动作进行采样，推理时直接取mean值。  

4. 参考代码  
   [gymnasium官方](https://gymnasium.org.cn/tutorials/training_agents/reinforce_invpend_gym_v26/)


## 训练结果
- CartPole-v1  
![CartPole-v1 train reward](https://github.com/iLovEing/hello_DRL/blob/main/2.REINFORCE/train_log/CartPole-v1_REINFORCE_reward.png)

- InvertedPendulum-v5  
![InvertedPendulum-v5 train reward](https://github.com/iLovEing/hello_DRL/blob/main/2.REINFORCE/train_log/InvertedPendulum-v5_REINFORCE_reward.png)
