# Import some modules from other libraries
import numpy as np
import torch
import time
import random
import math
import matplotlib.pyplot as plt
# Import the environment module
from environment import Environment
from collections import deque
import cv2
from q_value_visualiser import QValueVisualiser

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None

        #self.action_set = {0,1,2,3}

        self.epsilon = 1
        #self.epsilon_decay = 0.993
        #self.epsilon_min = 0.05

        self.eps_start = 1
        self.eps_end = 0.05
        self.eps_decay = 100

        self.goal_hits = 0
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0
        #self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        self.goal_hits = 0

    # Function to make the agent take one step in the environment.
    def step(self, q_network):
        # Choose the next action.
        discrete_action = self._choose_next_action(q_network)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        if distance_to_goal == 0:
            self.goal_hits += 1
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self, q_network):
        q_network.eval()
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.array([0,1,2,3]))
        else:
            with torch.no_grad():
                action = q_network.forward(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)).max(1)[1].item()
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([0.1 ,0], dtype=np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        elif discrete_action == 3:
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))  #
        #if distance_to_goal == 0:
         #  reward = 10
        #else:
        #    reward = -1
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=5000)

    def add_transition(self, transition):
        self.buffer.append(transition)

    def sample_minibatch(self, minibatch_size):
        #batch = random.sample(self.buffer, minibatch_size)

        buffer_indices = np.random.choice(len(self.buffer), minibatch_size, replace=False)
        batch = [self.buffer[index] for index in buffer_indices]
        return batch


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        self.learning_rate = 0.001

        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        #target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions):
        #online learning
        #reward = torch.tensor(transition[2], dtype=torch.float32)
        #state = torch.tensor(transition[0], dtype=torch.float32).unsqueeze(0)  #extra dimension for mini batch

        #mini batch size n
        s, a, r, ns = zip(*transitions)
        states = torch.tensor(s, dtype=torch.float32) #1 dimensional tensors
        actions = torch.tensor(a, dtype=torch.int64)
        rewards = torch.tensor(r, dtype=torch.float32)
        next_states = torch.tensor(ns, dtype=torch.float32)

        #self.q_network.train()
        #self.target_network.eval()
        with torch.no_grad():
            #target network
            #predicted_next_q = self.target_network.forward(next_states).detach().max(1)[0].unsqueeze(1)    #gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

            #DDQN using target network to predict best actions and q network to predict Q-values for those actions
            predicted_next_q_all = self.target_network.forward(next_states).detach()
            predQ = self.q_network.forward(next_states).detach()
        max_actions = predicted_next_q_all.argmax(1)
        predicted_next_q = predQ.gather(dim=1, index=max_actions.unsqueeze(-1))

        #minibatch size n -> n predictions -> index by n actions (gather predictions using each action to index each prediction)
        predicted_q = self.q_network.forward(states).gather(dim=1, index=actions.unsqueeze(-1))

        #online learning minibatch size 1
        #predicted_q = self.q_network.forward(state)[0, transition[1]]

        loss = torch.nn.MSELoss()(predicted_q, (rewards.unsqueeze(1) +0.9*predicted_next_q))   #(rewards.unsqueeze(1) + 0.9*predicted_next_q)
        return loss
        # TODO

#function to compare paramters of q network and target network to check target network is only updated when intended
    def compare_models(self):
        models_differ = 0
        for key_item_1, key_item_2 in zip(self.q_network.state_dict().items(), self.target_network.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
        return(models_differ)


# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    rb = ReplayBuffer()

    losses = []
    episodes = []
    hits = []

    fig, ax = plt.subplots()
    ax.set(xlabel='Number Episodes', ylabel='Loss', title='Loss Curve for DQN with Target Network')
    episode= 0
    # Loop over episodes
    for episode in range(600):
        # Reset the environment for the start of the episode.
        agent.reset()
        #agent.epsilon = max(agent.epsilon*agent.epsilon_decay, agent.epsilon_min)
        agent.epsilon = agent.eps_end + (agent.eps_start - agent.eps_end)* math.exp(-1. * episode / agent.eps_decay)

        #episode+=1
        ep_losses = 0
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(50):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step(dqn.q_network)
            #add transition to replay biffer
            rb.add_transition(transition)

            #sample from replay buffer and calculate average loss of predictions
            if len(rb.buffer)>=100:
                minibatch = rb.sample_minibatch(100)
                loss = dqn.train_q_network(minibatch)
                ep_losses+=loss

        #average losses calculated at all steps in episode to get approimate loss for episode
        losses.append(ep_losses/50)
        episodes.append(episode)

        #number of times agent hits goal state in episode
        hits.append(agent.goal_hits)

        #update target network weights
        if episode%10==0:
            dqn.update_target()
        #print(episode, ':', dqn.compare_models())

        if episode%50==0:
            print(f'Episode: {episode}   Epsilon: {agent.epsilon}')

    print('hits: ', hits)

        # Plot and save the loss vs iterations graph
    ax.plot(episodes, losses, color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("loss_vs_episodes.png")


 #Visualize value function
    #list of x,y positions unifromly across the 1x1 grid
    sl = []
    for y in np.arange(0.95, -0.05, -0.1):
        for x in np.arange(0.05, 1.05, 0.1):
            sl.append(np.array([x,y]))
            #Q[(x, y)] = dqn.q_network.forward(torch.tensor(np.array((x,y)), dtype=torch.float32).unsqueeze(0))   #[0, transition[1]]
    #print(sl)

    #use q network to get Q values for s,a pairs
    dqn.q_network.eval()
    with torch.no_grad():
        q_values = dqn.q_network.forward(torch.tensor((sl), dtype=torch.float32))
    #print(q_values)

    # Q value dictionary to pass to visualisation function
    Q = {}
    for i, s in enumerate(sl):
        c = np.around(s, decimals=2)
        #Q[(round(s[0],2), round(s[1],2))] = q_values[i].detach().numpy()
        Q[(c[0], c[1])] = q_values[i].detach().numpy()

    #for k, v in Q.items():
        #print(k, v)

    # Create a visualiser
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    # Draw the image
    visualiser.draw_q_values(Q)

    #Draw Greedy Policy
    print('policy')
    path_list = []
    state_a = environment.init_state
    dqn.q_network.eval()
    for step_num in range(20):

        with torch.no_grad():
            action = dqn.q_network.forward(torch.tensor(state_a, dtype=torch.float32).unsqueeze(0)).max(1)[1].item()

        new_state = state_a + agent._discrete_action_to_continuous(action)
        new_state = np.around(new_state, decimals=2)
        if not (new_state[0] <1 and new_state[0]>0) or not(new_state[1] <1 and new_state[1]>0):
            print('action out of bounds')
            break
        path_list.append([state_a, new_state])
        state_a = new_state

    environment.draw_path(path_list)
