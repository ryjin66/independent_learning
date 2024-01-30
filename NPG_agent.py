import torch.nn as nn
from core import *


class Replay_buffer():

    # self.storage to store samples，max_size is the maximum number of samples
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    # add a sample to the set
    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    # sampling，batch_size
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, u, r, y = [], [], [], []
        for i in ind:
            X, U, R, Y = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            y.append(np.array(Y, copy=False))
        return np.array(x), np.array(u), np.array(r), np.array(y)


def fisher_vector_product(actor, states, p):
    p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)  # check kl_grad == 0

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p


# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(actor, states, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, sigma):
        super(Actor, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.sigma = sigma
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mu = self.fc4(x)
        std = torch.tensor(self.sigma, dtype=torch.float)
        logstd = torch.log(std)
        return mu, std, logstd

    def get_log_prob(self, s, a):
        mu, std, logstd = self.forward(s)
        var = std.pow(2)
        log_den = -(a - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
        return log_den.sum(1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, num_s, num_a, num_hidden):
        super(Critic, self).__init__()
        self.state_dim = num_s
        self.action_dim = num_a
        self.fc1 = nn.Linear(num_s + num_a, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combine s and a
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        q = self.fc4(x)
        return q


class agent():
    def __init__(self, d_s, d_a, sigma, gamma, p_learning_rate, q_learning_rate, buffer_size, batch_size):
        self.actor = Actor(d_s, d_a, 64, sigma)
        self.critic = Critic(d_s, d_a, 64)
        self.replay_buffer = Replay_buffer(buffer_size)
        self.batch_size = batch_size
        self.p_learning_rate = p_learning_rate
        self.q_learning_rate = q_learning_rate
        self.gamma = gamma
        self.sigma = sigma
        self.d_s = d_s
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.q_learning_rate)
        self.loss_criterion = nn.MSELoss()

    def action_step(self, s, is_test=False):
        mu, std, _ = self.actor(torch.Tensor(s).unsqueeze(0))
        if is_test:
            action = torch.normal(mu, 0.00001)
        else:
            action = torch.normal(mu, std)
        return action.data.numpy()

    def policy_update(self):
        s, a, r, s_ = self.replay_buffer.sample(self.batch_size)
        t_s = torch.tensor(s, dtype=torch.float)
        t_a = torch.tensor(a, dtype=torch.float)
        Q_s_a = self.critic(t_s, t_a)

        # Calculate the derivative of J(\theta)
        Q_s_a = torch.squeeze(Q_s_a.detach())
        log_p = self.actor.get_log_prob(t_s, t_a)
        J = torch.mean(log_p * Q_s_a)
        grad_J = torch.autograd.grad(J, self.actor.parameters())
        grad_J = flat_grad(grad_J)
        # Natural policy gradient
        step_dir = conjugate_gradient(self.actor, t_s, grad_J.data, nsteps=10)
        new_params = flat_params(self.actor) + self.p_learning_rate * step_dir
        update_model(self.actor, new_params)

    def critic_update(self):
        s, a, r, s_ = self.replay_buffer.sample(self.batch_size)
        a_ = self.action_step(s_)
        t_s = torch.tensor(s, dtype=torch.float)
        t_a = torch.tensor(a, dtype=torch.float)
        t_r = torch.tensor(r.reshape(r.shape[0], 1), dtype=torch.float)
        t_s_ = torch.tensor(s_, dtype=torch.float)
        t_a_ = torch.tensor(a_, dtype=torch.float)
        Q_s_a = self.critic(t_s, t_a)
        next_Q = t_r + self.gamma * self.critic(t_s_, t_a_)
        next_Q = next_Q.detach()
        Q_loss = self.loss_criterion(Q_s_a, next_Q)
        Q_loss = Q_loss.mean()
        self.critic_optimizer.zero_grad()
        Q_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
