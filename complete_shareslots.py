import numpy as np
import random
from DQN1 import DQNAgent
import pickle
import sys
import env2
import env3
import env5

print("Starting the test")
def intent_module(dialouge = None):
    '''
    Dummy intent module
    :param dialouge: The dialouge to judge intent
    :return: THe intent class
    'flight' : 0
    'airfare' : 1
    'airline' : 2
    'ground_service' : 3
    'ground_fare' : 4
    NO : 5

    '''
    if dialouge :
        # check if it contains a no or no, then terminate
        # if " no " in dialouge:
        #     return 5
        # if " no." in dialouge:
        #     return 5
        if "no" in dialouge:
            return 5

        if "No" in dialouge:
            return 5
        return random.randint(0,4) # return a random intent for now
    return random.randint(0,4) # return a random intent



# the following function will invoke the intent modeule and return the intent state
def get_intent(current_intent, user_dialouge=None):
    '''

    :param current_intent:  The previosu intent that was there
    :param user_dialouge: The dialouge the user will say, currently none, as no dialouges are there
    :return: The modified intent
    '''
    intent = intent_module(user_dialouge)
    if intent == 5:
        # the user wants to quit
        current_intent = np.zeros((5))
        return current_intent
    if current_intent[intent] == 1:
        return get_intent(current_intent, user_dialouge)
    else:
        current_intent = np.zeros((5))
        current_intent[intent] = 1
        return current_intent

# todo : we will need a function to transfer the state information from one intent to the other


all_slots = ['to_loc','from_loc','date','time', 'class', 'round_trip' , 'city_name', 'transport_type' ]
intent_slots = dict()
intent_slots['flight'] = ['to_loc','from_loc','date','time','class']
intent_slots['airfare'] = ['to_loc','from_loc','date','class','round_trip']
intent_slots['airline'] = ['to_loc','from_loc','class']
intent_slots['ground_service'] = ['city_name', 'transport_type']
intent_slots['ground_fare'] = ['city_name', 'transport_type']


# slot values for all possible slots


complete_state = dict()
# init all slots to zero
for slot in all_slots:
    complete_state[slot]  = 0
    
def check_intent(inte,k):

    t=0

    for i in range(0,len(inte)):
    	if(inte[i]!=k):
    		t=t+1
    if(t==len(inte)):
    	return 1
    else:
    	return 0


def context_switcher(state, previous_intent, next_intent, complete_state, intent_slots):
    '''
    This function first need to update the global memory storage for a slot then retrieve the slots
    appropriate for the switching context and then return that state
    This function will help us in the context switch of the
    :param state : The state of the previos intent
    :param previous_intent: The tag for the previous intent
    :param next_intent:  The tag of the upcoming intent
    :param all_slots: The global storage of all intents
    :param intent_slots: The dictionary mapping for different intents and there slots
    :return: The state to be used for the next intent
    '''
    # first update the global memory with the new slots values
    #todo, id the state has a slot which has a lower confidence for a slot than the global memory what are we supposed to do then
    for i,slot in enumerate(intent_slots[previous_intent]):
        # for the slots in the previous intent update the global memory
        # fixme To check with the global memory thing as mentioned in the above todo
        complete_state[slot] = max(state[i], complete_state[slot])

    # get the next state
    next_state = np.zeros((len(intent_slots[next_intent])))
    # update the next sate with the appropriate values for the slot
    for i, slot in enumerate(intent_slots[next_intent]):
        next_state[i] = complete_state[slot]


    return next_state


# we will load all the models

# intents are followings
# 1. Flight: to_loc,from_loc,date,time,class
# 2. Airfare : to_loc, from_loc, class, date, round_trip
# 3.  Airline : to_loc, from_loc, class
# 4. Ground_service : city_name, transport_type
# 5. Ground_fare : city_name, transport_type

intents = ['flight','airfare','airline','ground_service','ground_fare']
# intent details will contain propertires like intent state size and intent action size

intent_details = dict()
intent_details['flight'] = [5, 13]
intent_details['airfare'] = [5, 13]
intent_details['airline'] = [3, 8]
intent_details['ground_service'] = [2, 6]
intent_details['ground_fare'] = [2, 6]

models = dict()
file2 = './save2/ENVORIGINAL_2018-08-25 20_36_33.613201_[75]_0.0_0.05_0.7_relu_50000.h5'
file3 = './save3/model3.h5'
file5 = './save5/ENVORIGINAL_2018-08-25 20_38_49.561754_[75]_0.0_0.05_0.7_relu_50000.h5'
models[2] = file2
models[3] = file3
models[5] = file5
# above the files that contain the repestive models for each kind of the state we are experimenting with
# load each of the indiviudla models
agents = dict()
# agents for each of the state size
# for each intent we will init a module of the type of agent
envs = dict() # env with differet state size and action size
# now make an env for each intent
for i in intents:
    if intent_details[i][0] == 3:
        envs[i] = env3.DialougeSimulation()
    elif intent_details[i][0] == 5:
        envs[i] = env5.DialougeSimulation()
    elif intent_details[i][0] == 2:
        envs[i] = env2.DialougeSimulation()
    else:
        "Wrong Configuration Inccured"
        raise Exception("Wrong Config")

# load the agent for each intent
for i in intents:
    print("Making Agent {}".format(i))
    # load the type of agent for each intent type
    agents[i] = DQNAgent(envs[i].state_size, envs[i].actions, hiddenLayers= [75], dropout= 0.0 , activation = 'relu',loadname = models[envs[i].state_size], saveIn = False, learningRate = 0.5, discountFactor=0.7, epsilon=0.01)


Episodes=100
rewards = []
iteration=[]

for e in range(Episodes):
    print("new episode====================")
# probability of ending a dialouge with no as the answer
    probability = 0.6
# The intent, variable to store the intento
# need to modify to take arbitary number of intents
    current_intent = np.zeros((5))
# current_intent = current_intent*-1
# currently no intent
    inte=[] 						#why????
# get the first intent
    current_intent = get_intent(current_intent)
    k=np.argmax(current_intent)
	#print(k)
    inte.append(k)
    print(inte)
    total_reward_dialouge = 0
    reward_intent = []
    intent_encountered = []
    intent_iterations = []
# when the system starts :
    intent_tag = np.argmax(current_intent)
    intent = intents[intent_tag]
    s = envs[intent].state_size
    a = envs[intent].actions
# start the state with all zeros

    state = np.zeros((s)) # the state size
    envs[intent].current_state = state # assign the current state of the env as the appropriate state
    previous_intent = intent
    print(previous_intent)
    while np.sum(current_intent) != 0:

    	intent_tag = np.argmax(current_intent)
    	next_intent = intents[intent_tag]
    	print(next_intent)
    	state = context_switcher(state,previous_intent,next_intent,complete_state,intent_slots)
    	print("Switched to {}".format(next_intent))
    	s = envs[next_intent].state_size
    	a = envs[next_intent].actions
    	envs[next_intent].current_state = state # assign the env current state as this state
    	stateOriginal = state.copy()
    	all_act = []
    	for z in range(0, a):
        	all_act.append(z)
    	intent_reward = 0 # the reward for the current intent
    	done = False
    	iter = 0
    	state = np.reshape(state, [1, len(state)])

    	while not done:
        	print("State : {}".format(state))
        	action = agents[next_intent].act(state, all_act)
        	print("Action : {}".format(action))

        	next_state, reward, done = envs[next_intent].step(action)
        	next_state = np.reshape(next_state , [s])
        	next_state = np.reshape(next_state, [1, s])
        	next_stateOriginal = next_state.copy()
        	state = next_state
        	stateOriginal = next_stateOriginal.copy()
        	intent_reward += reward
        	iter +=1

    	intent_encountered.append(next_intent)
    	total_reward_dialouge += intent_reward
    	reward_intent.append(intent_reward)
    	intent_iterations.append(iter)
    	print(reward_intent)
    	print(intent_iterations)
    	if(len(inte)<1):
        	print("Getting the next intent")
        	current_intent = get_intent(current_intent)
        	previous_intent = next_intent
        	k=np.argmax(current_intent)
        	t=check_intent(inte, k)
        	while(t!=1):
        		current_intent = get_intent(current_intent)
        		k=np.argmax(current_intent)
        		t=check_intent(inte, k)
        
        	inte.append(k)
        	print(inte)	
        	state = np.reshape(state, [s])
    	else:
        	print("Breaking from the dialouge")
        	rewards.append(np.sum(reward_intent,axis=0))
        	iteration.append(np.sum(intent_iterations,axis=0))
        	#state = np.zeros((s))
        	for slot in all_slots:
    			complete_state[slot]  = 0
	   # the break in the conversation
        	break


    print(rewards)
    print(iteration)


print(np.mean(rewards,axis=0))
print(np.std(rewards,axis=0))
print(np.mean(iteration,axis=0))
print(np.std(iteration,axis=0))











