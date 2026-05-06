import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Easy21():

    def __init__(self):
        self.minCardValue, self.maxCardValue = 1, 10
        self.dealerUpperBound = 17 # dealer always hits till sum is 17 or lesser
        self.gameLowerBound, self.gameUpperBound = 0, 21

    @classmethod
    def actionSpace(self):
        return (0, 1)

    def initGame(self):
        return (np.random.randint(self.minCardValue, self.maxCardValue+1),
               np.random.randint(self.minCardValue, self.maxCardValue+1))

    def draw(self):
        value = np.random.randint(self.minCardValue, self.maxCardValue+1)

        if np.random.random() <= 1/3: # positive value for black with probability 2/3, negative for red with probability 1/3
            return -value
        else:
            return value

    def step(self, playerValue, dealerValue, action):

        assert action in [0, 1], "Expected action in {0, 1} but got %i"%action

        if action == 0:

            playerValue += self.draw()

            # check if player busted
            if not (self.gameLowerBound < playerValue <= self.gameUpperBound):
                reward = -1
                terminated = True

            else:
                reward = 0
                terminated = False

        elif action == 1:
            terminated = True

            while self.gameLowerBound < dealerValue < self.dealerUpperBound:
                dealerValue += self.draw()

            # check if dealer busted // playerValue greater than dealerValue
            if not (self.gameLowerBound < dealerValue <= self.gameUpperBound) \
                or playerValue > dealerValue:
                reward = 1

            elif playerValue == dealerValue:
                reward = 0

            elif playerValue < dealerValue:
                reward = -1

        return playerValue, dealerValue, reward, terminated

class transitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [player_sum/21, dealer_sum/10, action]
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Regression heads
        self.next_p = nn.Linear(64, 1)   # normalized to [0, 1] then scaled by 21
        self.next_d = nn.Linear(64, 1)   # normalized to [0, 1] then scaled by 10

        # Classification heads
        self.reward = nn.Linear(64, 3)   # classes: 0,1,2 -> rewards -1,0,1
        self.terminal = nn.Linear(64, 1) # logit for terminal probability

    def forward(self, x):
        h = self.fc(x)
        p_next = self.next_p(h)
        d_next = self.next_d(h)
        r_logits = self.reward(h)
        t_logit = self.terminal(h)
        return p_next, d_next, r_logits, t_logit

class NNRecursiveWrapper:
    def __init__(self, real_env, a=0.1, b=0.5, buffer_size=2000, lr=1e-3):
        self.env = real_env
        self.a = a
        self.b = b
        self.c = 1.0 - b

        self.buffer = []
        self.next_buffer = []
        self.use_next_buffer = False
        self.buffer_size = buffer_size

        self.model = transitionNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def _clip_player(self, value):
        return int(np.clip(value, 1, 21))

    def _clip_dealer(self, value):
        return int(np.clip(value, 1, 10))

    def train_model(self, epochs=50, batch_size=128, w_p=1.0, w_d=1.0, w_r=1.0, w_t=1.0):
        if len(self.buffer) == 0:
            return []

        self.model.train()
        n = len(self.buffer)
        epoch_losses = []

        for _ in range(epochs):
            # Shuffle each epoch
            idxs = np.random.permutation(n)
            batch_losses = []

            for start in range(0, n, batch_size):
                bidx = idxs[start:start + batch_size]
                batch = [self.buffer[i] for i in bidx]

                x = torch.tensor(
                    [[p / 21.0, d / 10.0, float(a)] for (p, d, a, _, _, _, _) in batch],
                    dtype=torch.float32
                )
                y_p = torch.tensor([[p_n / 21.0] for (_, _, _, p_n, _, _, _) in batch], dtype=torch.float32)
                y_d = torch.tensor([[d_n / 10.0] for (_, _, _, _, d_n, _, _) in batch], dtype=torch.float32)
                y_r = torch.tensor([r + 1 for (_, _, _, _, _, r, _) in batch], dtype=torch.long)
                y_t = torch.tensor([[float(t)] for (_, _, _, _, _, _, t) in batch], dtype=torch.float32)

                pred_p, pred_d, r_logits, t_logit = self.model(x)

                loss_p = self.mse(pred_p, y_p)
                loss_d = self.mse(pred_d, y_d)
                loss_r = self.ce(r_logits, y_r)
                loss_t = self.bce(t_logit, y_t)

                loss = w_p * loss_p + w_d * loss_d + w_r * loss_r + w_t * loss_t

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

        return epoch_losses

    def begin_next_buffer(self):
        self.next_buffer = []
        self.use_next_buffer = True

    def commit_next_buffer(self):
        if self.use_next_buffer:
            self.buffer = self.next_buffer
            self.next_buffer = []
            self.use_next_buffer = False

    def discard_next_buffer(self):
        self.next_buffer = []
        self.use_next_buffer = False

    def step(self, p, d, action):
        roll = np.random.random()

        if roll < self.a:
            p_next, d_next, r, term = self.env.step(p, d, action)
        else:
            sub_roll = np.random.random()

            if sub_roll < self.b:
                self.model.eval()
                with torch.no_grad():
                    inp = torch.tensor([[p / 21.0, d / 10.0, float(action)]], dtype=torch.float32)
                    p_out, d_out, r_logits, t_logit = self.model(inp)

                    p_next = self._clip_player(p_out.item() * 21.0)
                    d_next = self._clip_dealer(d_out.item() * 10.0)
                    r = int(torch.argmax(r_logits, dim=1).item()) - 1
                    term = bool(torch.sigmoid(t_logit).item() > 0.5)
            else:
                if len(self.buffer) > 0:
                    idx = np.random.randint(len(self.buffer))
                    _, _, _, p_next, d_next, r, term = self.buffer[idx]
                else:
                    p_next, d_next, r, term = self.env.step(p, d, action)

        p_next = self._clip_player(p_next)
        d_next = self._clip_dealer(d_next)

        self._update_buffer(p, d, action, p_next, d_next, r, term)
        return p_next, d_next, r, term

    def _update_buffer(self, p, d, a, p_n, d_n, r, t):
        target_buffer = self.next_buffer if self.use_next_buffer else self.buffer
        target_buffer.append((p, d, a, p_n, d_n, r, t))
        if len(target_buffer) > self.buffer_size:
            target_buffer.pop(0)
