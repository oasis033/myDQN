import cv2
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from gameTRY import Breakout
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.05
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 750000
        self.minibatch_size = 32
        self.explore = 3000000 # Timesteps to go from INITIAL_EPSILON to FINAL_EPSILON

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

def preprocessing(image):
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    image_tensor = image_data.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor).to(device)
    return image_tensor

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    game_state = Breakout()
    D = deque()
    action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
    action[0] = 0
    image_data, reward, terminal = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0).to(device)

    # epsilon = 0.08870
    # iteration = 660000
    # epsilon = 0.08370
    # iteration = 960000
    # epsilon = 0.07870
    # iteration = 1260000
    # epsilon = 0.07704
    # iteration = 1360000
    epsilon = 0.07106
    iteration = 1720000

    while iteration < model.number_of_iterations:
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
        random_action = random.random() <= epsilon
        if random_action:
            print("Random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int).to(device)
                        if random_action
                        else torch.argmax(output)][0]
        action[action_index] = 1

        if epsilon > model.final_epsilon:
            epsilon -= (model.initial_epsilon - model.final_epsilon) / model.explore

        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0).to(device)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0).to(device)
        D.append((state, action, reward, state_1, terminal))

        if len(D) > model.replay_memory_size:
            D.popleft()

        minibatch = random.sample(D, min(len(D), model.minibatch_size))
        state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(device)
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(device)
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).to(device)
        
        output_1_batch = model(state_1_batch)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch)))).to(device)
        
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()
        state = state_1
        iteration += 1

        if iteration % 20000 == 0:
            torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")
              
        print("total iteration: {} Elapsed time: {:.2f} epsilon: {:.5f} action: {} Reward: {:.1f}".format(iteration, ((time.time() - start) / 60), epsilon, action_index.cpu().detach().numpy(), reward.cpu().numpy()[0][0]))

def test(model):
    model.to(device)
    game_state = Breakout()
    action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
    action[0] = 1
    image_data, reward, terminal = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0).to(device)

    while True:
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
        action_index = torch.argmax(output)
        action[action_index] = 1
        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0).to(device)
        state = state_1

def main(mode):
    if mode == 'test':
        model = torch.load('trained_model/current_model_1720000.pth', map_location=device).eval()
        test(model)
    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        model = NeuralNetwork()
        model.apply(init_weights)
        start = time.time()
        train(model, start)
    elif mode == 'continue':
        model = torch.load('trained_model/current_model_1720000.pth', map_location=device).eval()
        start = time.time()
        train(model, start)

if __name__ == "__main__":
    #main('test')
    main('continue')
