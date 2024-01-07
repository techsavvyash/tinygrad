from tinygrad import Tensor, nn, Variable, TinyJit
import random
import gymnasium as gym
import numpy as np
from examples.gpt2 import TransformerBlock
from tinygrad.helpers import Timing
from tqdm import trange
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

MAX_CONTEXT = 1024

class TheStupidest:
  def __init__(self, state_space, action_space):
    self.l1 = nn.Linear(state_space, 128)
    self.l2 = nn.Linear(128, action_space)
  def __call__(self, x):
    ret = self.l1(x).relu()
    return self.l2(ret)

"""
class EvenStupiderDecisionTransformer:
  def __init__(self, state_space, action_space):
    self.a_embedding = nn.Embedding(action_space+2, 1)
    self.layer1 = nn.Linear(4, action_space+1)

  def __call__(self, R:Tensor, s:Tensor, a:Tensor, start_pos:Variable):
    ae = self.a_embedding(a)
    tok_emb = Tensor.cat(ae, R, s, dim=-1)
    print(tok_emb.numpy())
    return self.layer1(tok_emb)
"""

class StupidDecisionTransformer:
  def __init__(self, state_space, action_space, dim=12):
    self.s_embedding = nn.Linear(state_space, dim//3)
    self.a_embedding = nn.Embedding(action_space+2, dim//3)
    self.R_embedding = nn.Linear(1, dim//3)
    #self.forward_jit = TinyJit(self.forward)
    self.layer1 = nn.Linear(dim, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, action_space+1)
    self.dim = dim

  def forward(self, R:Tensor, s:Tensor, a:Tensor, start_pos:Variable):
    #if Tensor.training:
    #  print(R.numpy())
    #  print(s.numpy())
    #  print(a.numpy())
    bs, seqlen = R.shape[0], R.shape[1]
    ae = self.a_embedding(a)
    re = self.R_embedding(R)
    se = self.s_embedding(s)
    #print(R.numpy())
    #print(re.numpy())
    tok_emb = Tensor.cat(ae, re, se, dim=-1).reshape(bs, seqlen, self.dim)
    #print(tok_emb.numpy())
    ret = self.layer1(tok_emb).relu()
    ret = self.layer2(ret).relu()
    ret = self.layer3(ret)
    #if Tensor.training:
    #  print(ret.exp().softmax().numpy())
    return ret

  def __call__(self, R, s, a, start_pos):
    return self.forward(R, s, a, start_pos)

class DecisionTransformer:
  def __init__(self, state_space, action_space, dim=384, n_heads=6, n_layers=4, norm_eps=1e-5, max_seq_len=256):
    self.dim = dim
    self.s_embedding = nn.Linear(state_space, dim//3)
    self.a_embedding = nn.Embedding(action_space+2, dim//3)
    self.R_embedding = nn.Linear(1, dim//3)
    self.wpe = nn.Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.ln_f = nn.LayerNorm(dim, norm_eps)
    self.lm_head = nn.Linear(dim, action_space+1, bias=True)
    self.forward_jit = TinyJit(self.forward)

  # [a_t-1, r_t, s_t] = 1 token
  def forward(self, R:Tensor, s:Tensor, a:Tensor, start_pos:Variable):
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
    #print(R.shape, s.shape, a.shape)
    bs, seqlen = R.shape[0], R.shape[1]
    ae = self.a_embedding(a)
    re = self.R_embedding(R)
    se = self.s_embedding(s)
    tok_emb = Tensor.cat(ae, re, se, dim=-1).reshape(bs, seqlen, self.dim)
    pos_emb = self.wpe(self.allpos.shrink((None, (start_pos, start_pos+seqlen))))
    h = tok_emb + pos_emb
    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None
    for hi in self.h: h = hi(h, start_pos, mask)
    logits = self.lm_head(self.ln_f(h))
    return logits.realize()

  def __call__(self, R:Tensor, s:Tensor, a:Tensor, start_pos:Variable):
    return (self.forward_jit if R.shape[1] == 1 else self.forward)(R, s, a, start_pos)
    #return self.forward(R, s, a, start_pos)

def sample(logits:Tensor, temperature:float=0.0):
  if temperature < 1e-6:
    ret = logits.argmax(-1)
  else:
    ret = (logits.exp()/temperature).softmax().multinomial()
  return ret.flatten().realize()

# compute max discounted RTG
#print(sum(np.power(0.95, np.arange(500))))

def rollout(env, model, desired_return):
  model.forward_jit.reset()
  Tensor.training = False
  Tensor.no_grad = True
  (obs, _), terminated, truncated = env.reset(), False, False
  target_return, cnt, act = desired_return, 0, n_act
  a, R, s = [n_act+1], [0], [obs]
  tc = 0
  while not terminated and not truncated:
    # NOTE: last action is a fake placeholder action
    a_logits = model(Tensor([[[target_return]]]), Tensor([[obs]]), Tensor([[act]]),
                    Variable("start_pos", 1 if cnt else 0, MAX_CONTEXT).bind(cnt))
    act = sample(a_logits[:, -1, :], 0.9).item()
    #print(act)
    if act == n_act:   # no idea
      tc += 1
      act = random.choice([0,1])
    obs, rew, terminated, truncated, _ = env.step(act)
    #print(terminated, truncated)
    cnt += 1
    target_return -= rew
    R.append(rew)
    s.append(obs)
    a.append(act)
    #print(f"{cnt:3d}: {act:d}")
  #discounts = np.power(0.99, np.arange(len(R)))
  #R = [np.sum(R[i:] * discounts[:len(R)-i]) for i in range(len(R))]
  R = [np.sum(R[i:]) for i in range(len(R))]
  print(f"{len(R):2d} rew={R[0]:.2f} twos={(tc/len(R)*100):.1f}% p(0):{(sum(np.array(a)==0)/len(a))*100:.2f} p(1):{(sum(np.array(a)==1)/len(a))*100:.2f}")
  return [[r] for r in R],s,a

def clip_r(R, s, a, K):
  R = R[:K]
  s = s[:K]
  a = a[:K]
  R += [[0] for _ in range(len(R), K)]
  s += [[0]*len(s[0]) for _ in range(len(s), K)]
  a += [-1 for _ in range(len(a), K)]
  return R, s, a

class PressTheLightUpButton: #(gym.Env):
  def __init__(self, size=2, game_length=1, hard_mode=False):
    self.size, self.game_length = size, game_length
    self.observation_space = gym.spaces.Box(0, 1, shape=(self.size,), dtype=int)
    self.action_space = gym.spaces.Discrete(self.size)
    self.step_num = 0
    self.done = True
    self.hard_mode = hard_mode

  def _new_state(self):
    self.n = random.randint(0, self.size-1)
    self.obs = [0]*self.size
    self.obs[self.n] = 1

  def reset(self):
    self.step_num = 0
    self.done = False
    self._new_state()
    return (self.obs, None)

  def step(self, act):
    if self.hard_mode:
      print(act, self.step_num)
      reward = self.obs[(act + self.step_num) % self.size]
    else:
      reward = self.obs[act]
    self.step_num += 1
    self._new_state()
    if not reward: self.done = True
    return self.obs, reward, self.done, self.step_num >= self.game_length, None

# works!
def stupidest_test():
  model = TheStupidest(3, 2)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-2)

  X = Tensor([[1.,1,0], [1,0,1], [0,1,0], [0,0,1]])
  Y = Tensor([[1.,0], [0,1], [0,1], [1,0]])

  for i in range(100):
    out = model(X)
    loss = (out-Y).square().mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())

# works!
def stupidest_test_2():
  model = TheStupidest(3, 2)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-2)

  X = Tensor([[1.,1,0], [1,0,1], [0,1,0], [0,0,1]])
  Y = Tensor([0, 1, 1, 0])

  for i in range(100):
    out = model(X)
    loss = out.sparse_categorical_crossentropy(Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())

# works!
def stupider_test():
  Tensor.training = True
  Tensor.no_grad = False

  model = StupidDecisionTransformer(2, 2)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)

  target_return = Tensor([[[1.]], [[1]], [[0]], [[0]]])
  state = Tensor([[[1.,0.]], [[0.,1.]], [[1.,0.]], [[0.,1.]]])
  fake_act = Tensor([[3.], [3], [3], [3]])
  desired_act = Tensor([[0.], [1], [1], [0]])

  print(target_return.shape)
  print(state.shape)
  print(fake_act.shape)

  for i in range(100):
    a_logits = model(target_return, state, fake_act, None)
    print(a_logits.numpy(), desired_act.numpy())
    loss = a_logits.sparse_categorical_crossentropy(desired_act)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())

  print("test model")
  for target_return in [1.,0.]:
    for desired_act, state in enumerate([[1.,0.], [0.,1.]]):
      if target_return == 0: desired_act = 1-desired_act
      a_logits = model(Tensor([[[target_return]]]), Tensor([[state]]), Tensor([[3]]), None)
      print(target_return, state, a_logits.log_softmax().exp().numpy(), a_logits.argmax().item(), desired_act)

def human_play(env):
  (obs, _) = env.reset()
  while 1:
    print(obs)
    act = int(input())
    s = env.step(act)
    print(s)
    obs = s[0]

if __name__ == "__main__":
  #stupidest_test()
  #stupidest_test_2()
  #stupider_test()
  #exit(0)

  plt.figure(2)

  env = gym.make('LunarLander-v2') #, render_mode='human')
  #env = gym.make('CartPole-v1') #, render_mode='human')
  #env = PressTheLightUpButton(size=2, game_length=32, hard_mode=False)
  #human_play(env)
  #exit(0)

  n_obs, n_act = int(env.observation_space.shape[0]), int(env.action_space.n)
  #model = StupidDecisionTransformer(n_obs, n_act)
  #optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)

  model = DecisionTransformer(n_obs, n_act)
  #optim = nn.optim.Adam(nn.state.get_parameters(model), lr=3e-5)
  #optim = nn.optim.Adam(nn.state.get_parameters(model), lr=3e-4)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)

  desired, returns, losses = [], [], []

  all_R, all_s, all_a = [], [], []
  BS = 128
  step = 0
  highest_reward = -1000
  while 1:
    if step%25 == 1: env = gym.make('LunarLander-v2', render_mode='human')
    else: env = gym.make('LunarLander-v2')

    """
    print("test model")
    Tensor.no_grad = True
    for i in range(n_obs):
      target_return = 1
      state = [0] * n_obs
      state[i] = 1
      desired_act = i
      a_logits = model(Tensor([[[target_return]]]), Tensor([[state]]), Tensor([[3]]), Variable("start_pos", 0, MAX_CONTEXT).bind(0))
      print(target_return, state, a_logits.log_softmax().exp().numpy(), a_logits.argmax().item(), desired_act)
    """

    """
    print("test model")
    Tensor.no_grad = True
    for target_return in [1.,0.]:
      for desired_act, state in enumerate([[1.,0.], [0.,1.]]):
        if target_return == 0: desired_act = 1-desired_act
        a_logits = model(Tensor([[[target_return]]]), Tensor([[state]]), Tensor([[3]]), Variable("start_pos", 0, MAX_CONTEXT).bind(0))
        print(target_return, state, a_logits.log_softmax().exp().numpy(), a_logits.argmax().item(), desired_act)
    """

    samps = 2
    while len(all_R) < BS//2 or samps > 0:
      #R,s,a = rollout(env, model, (highest_reward*(random.random()/5+0.8))+(abs(highest_reward)*random.random()*0.05)+(random.random()*0.1))
      R,s,a = rollout(env, model, highest_reward)
      returns.append(R[0])
      highest_reward = max(highest_reward, R[0][0])
      desired.append(highest_reward)
      R,s,a = clip_r(R,s,a,128)
      all_R.append(R); all_s.append(s); all_a.append(a)
      samps -= 1

    samples = Tensor.randint(BS, high=len(all_R)).numpy().tolist()
    br = Tensor(np.array(all_R, dtype=np.float32)[samples])
    bs = Tensor(np.array(all_s, dtype=np.float32)[samples])
    ba = Tensor(np.array(all_a, dtype=np.int32)[samples])

    #print(br.shape, bs.shape, ba.shape)

    Tensor.training = True
    Tensor.no_grad = False
    a_logits = model.forward(br[:, :-1], bs[:, :-1], ba[:, :-1], start_pos=Variable("start_pos", 0, MAX_CONTEXT).bind(0))
    #print(ba.numpy())
    #print("bs", bs[:, :-1].numpy())
    #print("ba", ba[:, 1:].numpy())
    #print("br", br[:, :-1].numpy())
    #print("logits", a_logits[:, :-1, :].log_softmax().exp().numpy())
    loss = a_logits.sparse_categorical_crossentropy(ba[:, 1:])
    losses.append(loss.item())
    print(f"loss = {loss.item():.3f}, highest_reward = {highest_reward:.2f}, samples = {len(all_R)}")
    optim.zero_grad()
    loss.backward()
    optim.step()

    plt.clf()
    plt.subplot(211)
    plt.plot(returns)
    plt.plot(desired)
    plt.subplot(212)
    plt.plot(losses)
    plt.pause(0.05)
    step += 1