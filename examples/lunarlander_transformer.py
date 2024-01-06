from tinygrad import Tensor, nn, Variable, TinyJit
import random
import gymnasium as gym
import numpy as np
from examples.gpt2 import TransformerBlock
from tinygrad.helpers import Timing
from tqdm import trange
import matplotlib.pyplot as plt

MAX_CONTEXT = 1024

class StupidDecisionTransformer:
  def __init__(self, state_space, action_space, dim=12):
    self.s_embedding = nn.Linear(state_space, dim//3)
    self.a_embedding = nn.Embedding(action_space+2, dim//3)
    self.R_embedding = nn.Linear(1, dim//3)
    self.forward_jit = TinyJit(self.forward)
    self.dim = dim
    self.layer = nn.Linear(12, action_space+1)

  def forward(self, R:Tensor, s:Tensor, a:Tensor, start_pos:Variable):
    bs, seqlen = R.shape[0], R.shape[1]
    ae = self.a_embedding(a)
    re = self.R_embedding(R)
    se = self.s_embedding(s)
    tok_emb = Tensor.cat(ae, re, se, dim=-1).reshape(bs, seqlen, self.dim)
    return self.layer(tok_emb)

  def __call__(self, R, s, a, start_pos): return self.forward(R, s, a, start_pos)

class DecisionTransformer:
  def __init__(self, state_space, action_space, dim=384, n_heads=12, n_layers=4, norm_eps=1e-5, max_seq_len=1024):
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

def sample(logits:Tensor, temperature:float=0.0):
  if temperature < 1e-6:
    ret = logits.argmax(-1)
  else:
    ret = (logits.exp()/temperature).softmax().multinomial()
  return ret.flatten().realize()

# compute max discounted RTG
#print(sum(np.power(0.95, np.arange(500))))

highest_reward = -1000
def rollout(env, model):
  global highest_reward
  model.forward_jit.reset()
  Tensor.training = False
  Tensor.no_grad = True
  (obs, _), terminated, truncated = env.reset(), False, False
  target_return, cnt, act = 1, 0, n_act
  a, R, s = [n_act+1], [0], [obs]
  tc = 0
  while not terminated and not truncated:
    # NOTE: last action is a fake placeholder action
    a_logits = model(Tensor([[[target_return]]]), Tensor([[obs]]), Tensor([[act]]),
                    Variable("start_pos", 1 if cnt else 0, MAX_CONTEXT).bind(cnt))
    print(a_logits[:, -1, :].exp().softmax().numpy())
    act = sample(a_logits[:, -1, :], 0.0).item()
    if act == n_act:   # no idea
      tc += 1
      act = random.choice([0,1])
    obs, rew, terminated, truncated, _ = env.step(act)
    cnt += 1
    #target_return -= rew
    R.append(rew)
    s.append(obs)
    a.append(act)
    #print(f"{cnt:3d}: {act:d}")
  #discounts = np.power(0.99, np.arange(len(R)))
  #R = [np.sum(R[i:] * discounts[:len(R)-i]) for i in range(len(R))]
  R = [np.sum(R[i:]) for i in range(len(R))]
  highest_reward = max(highest_reward, R[0])
  print(f"{len(R):2d} rew={R[0]:.2f} twos={(tc/len(R)*100):.1f}% p(0):{(sum(np.array(a)==0)/len(a))*100:.2f} p(1):{(sum(np.array(a)==1)/len(a))*100:.2f}")
  return [[r] for r in R],s,a

def clip_r(R, s, a, K):
  R = R[:K]
  s = s[:K]
  a = a[:K]
  R += [[0] for _ in range(len(R), K)]
  s += [[0]*len(s[0]) for _ in range(len(s), K)]
  a += [2 for _ in range(len(a), K)]
  return R, s, a

class PressTheLightUpButton: #(gym.Env):
  def __init__(self):
    self.size = 2
    self.observation_space = gym.spaces.Box(0, 1, shape=(self.size,), dtype=int)
    self.action_space = gym.spaces.Discrete(self.size)

  def reset(self):
    # which button lights up?
    self.n = random.randint(0, self.size-1)

    self.obs = [0]*self.size
    self.obs[self.n] = 1
    return (self.obs, None)

  def step(self, act):
    return self.obs, self.obs[act], True, False, None

if __name__ == "__main__":
  #env = gym.make('LunarLander-v2', render_mode='human')
  #env = gym.make('CartPole-v1') #, render_mode='human')
  #print(int(env.observation_space.shape[0]))
  env = PressTheLightUpButton()

  """
  while 1:
    (obs, _) = env.reset()
    print(obs)
    act = int(input())
    print(env.step(act))
  exit(0)
  """


  n_obs, n_act = int(env.observation_space.shape[0]), int(env.action_space.n)
  #model = DecisionTransformer(n_obs, n_act)
  model = StupidDecisionTransformer(n_obs, n_act)
  optim = nn.optim.Adam(nn.state.get_parameters(model)) #, lr=3e-4)

  returns, losses = [], []

  all_R, all_s, all_a = [], [], []
  BS = 16
  step = 0
  while 1:
    #if step%10 == 0: env = gym.make('LunarLander-v2', render_mode='human')
    #else: env = gym.make('LunarLander-v2')

    R,s,a = rollout(env, model)
    #returns.append(len(R))
    returns.append(R[0])
    R,s,a = clip_r(R,s,a,4)
    print(R)
    print(s)
    print(a)

    all_R.append(R); all_s.append(s); all_a.append(a)

    samples = Tensor.randint(BS, high=len(all_R)).numpy().tolist()
    #samples = [0]*BS
    br = Tensor(np.array(all_R, dtype=np.float32)[samples])
    bs = Tensor(np.array(all_s, dtype=np.float32)[samples])
    ba = Tensor(np.array(all_a, dtype=np.int32)[samples])

    #print(br.numpy())
    #print(bs.numpy())
    #print(ba.numpy())

    Tensor.training = True
    Tensor.no_grad = False
    a_logits = model.forward(br, bs, ba, start_pos=Variable("start_pos", 0, MAX_CONTEXT).bind(0))
    #print(ba.numpy())
    loss = a_logits[:, :-1, :].sparse_categorical_crossentropy(ba[:, 1:])
    losses.append(loss.item())
    print(f"loss = {loss.item():.3f}, highest_reward = {highest_reward:.2f}, samples = {len(all_R)}")
    optim.zero_grad(); loss.backward(); optim.step()

    plt.plot(returns)
    plt.plot(losses)
    plt.pause(0.05)
    step += 1