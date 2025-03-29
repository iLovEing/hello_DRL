
# A2C

## 算法原理
`A2C`属于actor critic类算法，融合了policy gradient + value evaluation思想，其目标和REINFORCE类似，训练actor网络指导agent行动，其中action value(或 state value)的估计由MC方法换成了神经网络(critic)，并且引入advantage概念：  
1. 策略梯度的目标函数求导 :   
$\nabla_{\theta }J(\theta ) = E_{S\sim \eta , a\sim \pi } [\nabla_{\theta } \ln \pi (A\mid S, \theta )q_{\pi }(S, A) ]$   ----- :one:  
2. 为了减小q估计的方差，这里引入一个'基准'，为S的函数，引入后梯度不变 :  
$\nabla_{\theta }J(\theta ) = E_{S\sim \eta , a\sim \pi } [\nabla_{\theta } \ln \pi (A\mid S, \theta )(q_{\pi }(S, A) - b(S)) ]$   ----- :two:  
这里，可以证明引入项期望为0，即不会对目标函数优化有影响。 $b(S)$ 可以计算出最优解析解，在实际工程应用中，通常取state value, 最终critic输出写为: $q_{\pi }(S, A) - v_{\pi}(S)$  
这一项即为RL中经常提到的advantage，从字面上理解，在某一个state选择action时，应该选择回报高于state value的action，这才是action所带来的'优势'。
3. 使用随机梯度代替随机变量, 同时, 上式中同时涉及到action value和state value，使用两个网络估计比较复杂，这里使用TD error表示advantage，即可用一个value network表示，得到A2C最终参数迭代式:  
  - TD error: $\delta (t) = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$  ----- :three:  
  - actior: $\theta_{t+1} = \theta_{t} + \alpha_\theta \delta_t \nabla_\theta ln\pi(a_t|s_t, \theta_t)$  ----- :four:  
  - critic: $w_{t+1} = w_{t} + \alpha_w \delta_t \nabla_w v(s_t, w_t)$  ----- :five:  
  其中, $w$ 为critic网络参数, $\theta$ 为actor网络参数，收集到的经验为 $(s_t, a_t, r_{t+1}, s_{t+1})$
4. **off policy A2C**, 目标函数中，action需要从策略 $\pi$ 中采样，因此A2C是**on-policy**算法。此时通过重要性采样可以将on-policy变为off-policy算法。  
    假设采样policy为 $\beta$ ,  目标policy为 $\pi$ , 结合重要性采样, 目标函数的导数变为:
    $\nabla_{\theta }J(\theta ) = E_{S\sim \rho  , A\sim \beta } [\frac{\pi (A\mid S, \theta )}{\beta (A\mid S)}  \nabla_{\theta } \ln \pi (A\mid S, \theta )q_{\pi }(S, A) ]$  ----- :six:  
    加入TD advantage，使用随机上升更新:  
    - TD error: $\delta (t) = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$  ----- :seven:  
    - actior: $\theta_{t+1} = \theta_{t} + \alpha_\theta \frac{\pi (a_t \mid s_t, \theta )}{\beta (a_t \mid s_t)} \delta_t \nabla_\theta ln\pi(a_t|s_t, \theta_t)$  ----- :eight:  
    - critic: $w_{t+1} = w_{t} + \alpha_w \frac{\pi (a_t \mid s_t, \theta )}{\beta (a_t \mid s_t)} \delta_t \nabla_w v(s_t, w_t)$   ----- :nine:  

## 代码说明
1. loss说明, 假设critic网络为 $v$ , actor网络为 $\pi$ , 收集经验的网络为 $\beta$ , 收集到的经验为 $(s_t, a_t, t_{t+1}, s_{t+1})$ . critic的目标为拟合advantage的前半部分，使用MSELoss； actor的损失函数可由式 :eight: 的梯度部分积分得到 (critic的损失同样可以从 :nine: 式积分得到，可以验证就是MSELoss ). **注意:** 这里带`~`上标表示只是系数，不带梯度流向，需要从计算图中detach.  
  - critic: $\frac{\tilde{\pi} (a_t \mid s_t, \theta )}{\tilde{\beta} (a_t \mid s_t)} ( r_{t+1} + \gamma \tilde{v} (s_{t+1}, w_t) - v(s_t, w_t))^2 $  
  - actor: $\frac{\tilde{\pi} (a_t \mid s_t, \theta )}{\tilde{\beta} (a_t \mid s_t)} (r_{t+1} + \gamma \tilde{v}(s_{t+1}, w_t) - \tilde{v}(s_t, w_t)) ln\pi(a_t|s_t, \theta_t)$  
   
3. 伪代码 (off-policy)
   1. 初始化：初始化replay buffer，随机初始化actor、critic  
   2. 循环采样、计算:
      1. 简化网络数量，仍然使用actor采样，但是保存prob, 可以用off-policy更新
      2. 采样得到一定数量的数据后，开始更新网络
      3. 计算advantage, 依次更新critic 和 actor 

4. tricks & discuss
  - 在GAN中，discriminator指导generator优化，通常会让前者的优化频率高于后者，这里借鉴这种做法，没优化k步critic，优化一次actor  
   


## 训练结果
- CartPole-v1  
![CartPole-v1 train reward](https://github.com/iLovEing/hello_DRL/blob/main/3.A2C/train_log/CartPole-v1_A2C_reward.png)

- CartPole-v1_off-policy
![CartPole-v1_off-policy train reward](https://github.com/iLovEing/hello_DRL/blob/main/3.A2C/train_log/CartPole-v1_A2C_off-policy_reward.png)

