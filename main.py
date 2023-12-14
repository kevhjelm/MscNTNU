#### Main file for our program #######

# Reference to go back to
# https://www.geeksforgeeks.org/q-learning-in-python/
# https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
# Epsilon greedy python reference
#https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/


#import libraries
import numpy as np
import json
import plotly.graph_objects as go # Using plotly was some problems using matplotlib, might be old VM
from detoxify import Detoxify #Detoxify library
from sklearn.metrics import f1_score 

# The possible Actions the moderator can take based on message toxicity
# ignore means "do nothing, the conversation is respectful"
# flag for review is considred grey zone
# Block or delete, the conversation is no longer respectful
ACTIONS = ["Ignore", "Flag for review", "Block or Delete"]



# Parameters for the Q-learning algorithm
# changes based on dataset, set to default
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1 # Exploration rate

# Implements of the Q-learning algorithm.

class QLearning:
    def __init__(self):
    # The Q-table being Itialized to a matrix of zeros, we are using states as rows and actions as columns
        self.q_table = np.zeros((100, len(ACTIONS)))

    def choose_action(self, state):
    # Decide on an action based on the current state using the epsilon-greedy policy

        if np.random.uniform(0, 1) < EPSILON:
        # With probability epsilon, choose a random action for exploration
            return np.random.choice(ACTIONS)
        else:
        # With the probability of (1-epsilon), The best action from Q-table for exploitation
            return ACTIONS[np.argmax(self.q_table[state, :])]
    # Update Q-table 
    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, ACTIONS.index(action)]
        target = reward + GAMMA * np.max(self.q_table[next_state, :])
                
        # Update the Q-value towards the target Q-value
        self.q_table[state, ACTIONS.index(action)] += ALPHA * (target - predict)
        
# Visualize the Q-table as a heatmap, image is also being saved
    def visualize_q_table(self):
        fig = go.Figure(data=go.Heatmap(z=self.q_table, x=ACTIONS, y=[f"State {i}" for i in range(100)], colorscale='Viridis'))
        fig.update_layout(title="Q-table Heatmap", xaxis_title='Actions', yaxis_title='States')
        fig.write_image("q_table_heatmap.png")

# creating our agents (users) and its properties
class User:
    def __init__(self, name):
        self.name = name #store the name of user
        self.prev_state = None # previous state
        self.current_state = None # set current state
        self.prev_action = None # previous action
        self.toxic_score = 0 # toxic score, starts at 0
        self.toxic_scores = [] #store all toxic score in list
        self.rewards = [] #store all rewards in list for each interaction
        self.cumulative_rewards = [] #store all cumulative  rewards in list
        self.current_cumulative_reward = 0 #Initialize  current cumulative  rewards

    # This function updates the user's state and score based on the latest interaction
    def update(self, state, action, toxic_score):
        self.prev_state = self.current_state
        self.current_state = state # Set the new current state
        self.prev_action = action # Update the previous action
        self.toxic_score = toxic_score # Update the current toxic score
        self.toxic_scores.append(toxic_score) # Add the new toxic score 
        
        # Calculating reward based on state: normal states give +1, medium 0, others (implied toxic) -1
        reward = 1 if state == "normal" else (0 if state == "medium" else -1)
        self.current_cumulative_reward += reward # Update the cumulative reward with the new reward
        self.cumulative_rewards.append(self.current_cumulative_reward)  # Append the updated cumulative reward
        self.rewards.append(ACTIONS.index(action)) # Add the index of the action taken to the rewards list

# Moderator class and tracking
class Moderator:
    def __init__(self):
        self.users = {"UserA": User("UserA"), "UserB": User("UserB")} #names our users
        self.general_toxic_scores = [] # track the overall toxic scores from all users
        self.conversation_log = [] # add conversation to a log
        self.learner = QLearning() #Qlearning to decide on actions
        
    # analyzing user input
    def get_response(self, username, user_input):
        result = Detoxify('original').predict(user_input) # Using Detoxify to predict toxicity of user input
        toxic_score = result['toxicity'] # Getting value from detoxify
        self.general_toxic_scores.append(toxic_score) # adding score to list

        state = int(toxic_score * 100) # Evaluates the state based on the toxic score
        action = self.learner.choose_action(state) # Chooses an action based on the current state

        # Determines the state name and reward based on the toxicity score
        if toxic_score < 0.3: 
            state_name = "normal" # Non-toxic state
            reward = 1 # Positive reward for normal state
        elif 0.3 <= toxic_score < 0.7:
            state_name = "medium" # Moderately toxic state
            reward = 0 # Neutral reward for moderate state
        else:
            state_name = "toxic" #  toxic state
            reward = -1 # Negative reward for toxic state

        next_state = int(np.mean(self.general_toxic_scores) * 100) # Calculates the next state based on the average toxicity
        self.learner.learn(state, action, reward, next_state) # The moderator learns from the current interaction

        self.users[username].update(state_name, action, toxic_score) # Updates the user's state and score
        
        # Formats the cumulative reward for display
        current_cuma = self.users[username].current_cumulative_reward
        cuma_sign = "+" if current_cuma >= 0 else ""
        
        # Constructs the response including the action taken, current state, and other details
        response = f"{username} Moderator (Action: {action}, State: {state_name}, Score: {toxic_score:.2f}, cuma: {cuma_sign}{current_cuma})"
        self.conversation_log.append({username: user_input, "Chatbot": response}) # Logs the conversation
        return response # Returns the moderator's response

    # function to save ALL statistics for future analysis, will be usefull
    def save_statistics(self):
        conversation_indices = list(range(1, len(self.conversation_log) + 1))

        for username, user in self.users.items():
            with open(f'{username}_toxic_scores.json', 'w') as f: # Save the user's toxic scores to a JSON file
                json.dump(user.toxic_scores, f, cls=NumpyEncoder)
            with open(f'{username}_cumulative_rewards.json', 'w') as f:  # Save the user's cumulative rewards to a JSON file
                json.dump(user.cumulative_rewards, f)
            with open(f'{username}_rewards.json', 'w') as f: # Save the user's list of rewards to a JSON file
                json.dump(user.rewards, f) 
                
                
            # Plot the user's toxic scores over interactions and save it as an image file, will be usefull
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=conversation_indices, y=user.toxic_scores, mode='lines+markers', name=f'{username} Toxic Score Over Interaction')) # markers/lines of dots or lines
            fig.update_layout(title=f"{username} Toxic Scores Over Interaction",
                              xaxis_title="Conversation #",
                              yaxis_title="Toxic Score")
            fig.write_image(f"{username}_toxic_scores_plot.png")
            
            fig = go.Figure()#drawing figures, adding titles, lines or dots etc
            fig.add_trace(go.Scatter(x=conversation_indices, y=user.cumulative_rewards, mode='lines+markers', name=f'{username} Cumulative Rewards Over Interaction'))
            fig.update_layout(title=f"{username} Cumulative Rewards Over Interaction", #image title
                              xaxis_title="Conversation #", #x-axis title
                              yaxis_title="Cumulative Reward") #y-axis title
            fig.write_image(f"{username}_cumulative_rewards_plot.png") # image of cumulativ reward

        with open('general_toxic_scores.json', 'w') as f:
            json.dump(self.general_toxic_scores, f, cls=NumpyEncoder)
        
        fig = go.Figure()#drawing figures, adding titles, lines or dots etc
        fig.add_trace(go.Scatter(x=conversation_indices, y=self.general_toxic_scores, mode='lines', name='General Toxic Score Over Interaction'))
        fig.update_layout(title="General Toxic Scores Over Interaction",
                          xaxis_title="Conversation #",
                          yaxis_title="Toxic Score")
        fig.write_image("general_toxic_scores_plot.png")
        
        with open('conversation_log.json', 'w') as f:
            json.dump(self.conversation_log, f)
        
        self.learner.visualize_q_table()
        
# Initialize Attempt to calculate FI scores. Need to store the true labels and predicted labels for all users
    def calculate_f1_scores(self):
        # Collect labels for all users
        all_true_labels = []
        all_predicted_labels = []
        for user in self.users.values(): # Loop through all user objects  compile true toxic scores
                    # Convert toxic scores to labels and extend the true labels list
            all_true_labels.extend([0 if score < 0.3 else (1 if score < 0.7 else 2) for score in user.toxic_scores])
            all_predicted_labels.extend(user.rewards)  # Extend the predicted labels list with the rewards

        # Calculate F1 score for each action
        f1_scores = {} # Create a dictionary to hold the F1 score for each action
        for i, action in enumerate(ACTIONS):
            # Create binary arrays for each action
            true_binary = [1 if label == i else 0 for label in all_true_labels]
            predicted_binary = [1 if label == i else 0 for label in all_predicted_labels]

            # Calculate F1 score for the action
            f1_scores[action] = f1_score(true_binary, predicted_binary, zero_division=1)

        return f1_scores
        
# function t o solve JSON encoder class to handle numpy float32 error
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj) # Convert np.float32 to native Python float type
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__": # Python main
    moderator = Moderator() #starting moterator object
    current_user = "UserA" #UserA always starts the conversation, doesnt really matter who starts
    while True:
        user_input = input(f"{current_user}: ")
        if user_input.lower() in ['exit', 'quit', 'end']: # we needed a way to exit the chat for the users
            moderator.save_statistics() # once out of chat, the save_statistics fuction will save it all
            f1_scores = moderator.calculate_f1_scores() # FI-scores will be calculated end of conversation
            print(f"F1-scores for actions: {f1_scores}")
            break
        response = moderator.get_response(current_user, user_input)# the conversation will go back and forth
        print(response)
        current_user = "UserB" if current_user == "UserA" else "UserA" # Adding fixed user 
