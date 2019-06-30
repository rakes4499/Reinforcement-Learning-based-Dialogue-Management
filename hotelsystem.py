import numpy as np
import random
from DQN1 import DQNAgent
import pickle
import sys
import env2
import env3
import env5
# import flightsystem
class hotel_system:
	print("in hotel system")
	def _init_(self):
		self.all_slots=[]
		self.intent_slots=dict()
		self.complete_state=dict()
		self.intents=[]
		self.intent_details=dict()
		self.models=dict()
		self.agents=dict()
		self.envs=dict()
		self.rewards=[]
		self.iteration=[]
		self.current_intent=np.zeros((5))
		self.inte=[]
		self.reward_intent = []
		self.intent_encountered = []
		self.intent_iterations =[]
		self.intent_tag=0
		self.intents=[]
		self.state=[]
		self.previous_intent=[]
		self.total_reward_dialouge=[]
		self.user_dialouge=None
		self.k=0
		self.done=False
	
	def intent_module(self,dialouge = None):
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
	        # if " no " in self.self.dialouge:
	        #     return 5
	        # if " no." in self.dialouge:
	        #     return 5
	        if "no" in dialouge:
	            return 5

	        if "No" in dialouge:
	            return 5
	        return random.randint(0,4) # return a random intent for now
	    return random.randint(0,0) # return a random intent



	# the following function will invoke the intent modeule and return the intent state
	def get_intent(self,current_intent,user_dialouge=None):
	    '''

	    :param current_intent:  The previous intent that was there
	    :param user_dialouge: The dialouge the user will say, currently none, as no self.self.dialouges are there
	    :return: The modified intent
	    '''

	    intent = self.intent_module(user_dialouge)
	    if intent == 5:
	        # the user wants to quit
	        current_intent = np.zeros((5))
	        return current_intent
	    if current_intent[intent] == 1:
	        return self.get_intent(current_intent, user_dialouge)
	    else:
	        current_intent = np.zeros((5))
	        current_intent[intent] = 1
	        return current_intent

	# # todo : we will need a function to transfer the state information from one intent to the other

	    
	def check_intent(self,inte,k):

	    t=0

	    for i in range(0,len(inte)):
	    	if(inte[i]!=k):
	    		t=t+1
	    if(t==len(inte)):
	    	return 1
	    else:
	    	return 0


	def context_switcher(self,state, previous_intent, next_intent, complete_state, intent_slots):
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

	def main(self,states):
		self.all_slots = ['to_loc','from_loc','date','n_adults','children' ]
		self.intent_slots = dict()
		self.intent_slots['book'] = ['to_loc','from_loc','date','n_adults','children']


	# slot values for all possible slots

		self.complete_state = dict()
	# init all slots to zero
		for slot in self.all_slots:
			self.complete_state[slot]  = 0
		self.intents = ['book']
	# intent details will contain propertires like intent state size and intent action size

		self.intent_details = dict()
		self.intent_details['book'] = [5, 13]

		self.models = dict()
		# self.file2 = './save2/ENVORIGINAL_2018-08-25 20_36_33.613201_[75]_0.0_0.05_0.7_relu_50000.h5'
		# self.file3 = './save3/model3.h5'
		self.file5 = './save5/ENVORIGINAL_2018-08-25 20_38_49.561754_[75]_0.0_0.05_0.7_relu_50000.h5'
		# self.models[2] = self.file2
		# self.models[3] = self.file3
		self.models[5] = self.file5
		# above the files that contain the repestive models for each kind of the state we are experimenting with
		# load each of the indiviudla models
		self.agents = dict()
		# agents for each of the state size
		# for each intent we will init a module of the type of agent
		self.envs = dict() # env with differet state size and action size
		# now make an env for each intent
		for i in self.intents:
			# if self.intent_details[i][0] == 3:
			# 	self.envs[i] = env3.DialougeSimulation()
			if self.intent_details[i][0] == 5:
				self.envs[i] = env5.DialougeSimulation()
			# elif self.intent_details[i][0] == 2:
			# 	self.envs[i] = env2.DialougeSimulation()
			else:
				"Wrong Configuration Inccured"
				raise Exception("Wrong Config")

		# load the agent for each intent
		for i in self.intents:
			print("Making Agent {}".format(i))
			# load the type of agent for each intent type
			self.agents[i] = DQNAgent(self.envs[i].state_size, self.envs[i].actions, hiddenLayers= [75], dropout= 0.0 , activation = 'relu',loadname = self.models[self.envs[i].state_size], saveIn = False, learningRate = 0.5, discountFactor=0.7, epsilon=0.01)


		self.Episodes=1
		self.rewards = []
		self.iteration=[]

		for e in range(self.Episodes):
			print("new episode====================")
		# probability of ending a dialouge with no as the answer
			self.probability = 0.6
		# The intent, variable to store the intento
		# need to modify to take arbitary number of intents
			self.current_intent = np.zeros((5))
		# current_intent = current_intent*-1
		# currently no intent
			self.inte=[] 						
		# get the first intent
			# self.current_intent = self.get_intent(self.current_intent)
			self.k=np.argmax(self.current_intent)
			#print(k)
			self.inte.append(self.k)
			print(self.inte)
			self.total_reward_dialouge = 0
			self.reward_intent = []
			self.intent_encountered = []
			self.intent_iterations = []
			self.current_intent = self.get_intent(self.current_intent)
		# when the system starts :
			self.intent_tag = np.argmax(self.current_intent)
			self.intent = self.intents[self.intent_tag]
			self.s = self.envs[self.intent].state_size
			self.a = self.envs[self.intent].actions
		# start the state with all zeros

			self.state = np.zeros((self.s)) # the state size
			self.envs[self.intent].current_state = self.state # assign the current state of the env as the appropriate state
			self.previous_intent = self.intent
			print(self.previous_intent)
			while np.sum(self.current_intent) != 0:

				self.intent_tag = np.argmax(self.current_intent)
				self.next_intent = self.intents[self.intent_tag]
				print(self.next_intent)
				self.state = self.context_switcher(self.state,self.previous_intent,self.next_intent,self.complete_state,self.intent_slots)
				print("Switched to {}".format(self.next_intent))
				self.s = self.envs[self.next_intent].state_size
				self.a =self.envs[self.next_intent].actions
				self.envs[self.next_intent].current_state = self.state # assign the env current state as this state
				self.stateOriginal = self.state.copy()
				self.all_act = []
				for z in range(0, self.a):
					self.all_act.append(z)
				self.intent_reward = 0 # the reward for the current intent
				self.done = False
				self.iter = 0
				self.state = np.reshape(self.state, [1, len(self.state)])

				while not self.done:
					print("State : {}".format(self.state))
					self.action = self.agents[self.next_intent].act(self.state,self.all_act)
					print("Action : {}".format(self.action))

					self.next_state, self.reward, self.done = self.envs[self.next_intent].step(self.action)
					self.next_state = np.reshape(self.next_state , [self.s])
					self.next_state = np.reshape(self.next_state, [1, self.s])
					self.next_stateOriginal = self.next_state.copy()
					self.state = self.next_state
					self.stateOriginal = self.next_stateOriginal.copy()
					self.intent_reward += self.reward
					self.iter +=1

				self.intent_encountered.append(self.next_intent)
				self.total_reward_dialouge += self.intent_reward
				self.reward_intent.append(self.intent_reward)
				self.intent_iterations.append(self.iter)
				print(self.reward_intent)
				print(self.intent_iterations)
				if(len(self.inte)<1):
					print("Getting the next intent")
					self.current_intent = self.get_intent(self.current_intent)
					self.previous_intent =self.next_intent
					self.k=np.argmax(self.current_intent)
					self.t=self.check_intent(self.inte, self.k)
					while(self.t!=1):
						self.current_intent = self.get_intent(self.current_intent)
						self.k=np.argmax(self.current_intent)
						self.t=self.check_intent(self.inte, self.k)
				
					self.inte.append(self.k)
					print(self.inte)	
					self.state = np.reshape(self.state, [self.s])
				else:
					print("Breaking from the dialouge")
					self.rewards.append(np.sum(self.reward_intent,axis=0))
					self.iteration.append(np.sum(self.intent_iterations,axis=0))
					#state = np.zeros((s))
					for slot in self.all_slots:
						self.complete_state[slot]  = 0
			# the break in the conversation
					break


			print(self.rewards)
			print(self.iteration)

			# self.m=[0,1]
			# zz=flightsystem.flight_system()
			# self.st=[0,0]
			# self.alt_st=[0,0,0,0,0]
			# self.st[0]=random.choice(self.m)
			# if self.st[0]==0:
			# 	# state[1]=random.choice(n)
			# 	zz.main1(self.alt_st)
			# else:
			# 	# main(alloted_states)
			# 	pass


		print(np.mean(self.rewards,axis=0))
		print(np.std(self.rewards,axis=0))
		print(np.mean(self.iteration,axis=0))
		print(np.std(self.iteration,axis=0))
		


	# we will load all the models

	# intents are followings
	# 1. Flight: to_loc,from_loc,date,time,class
	# 2. Airfare : to_loc, from_loc, class, date, round_trip
	# 3.  Airline : to_loc, from_loc, class
	# 4. Ground_service : city_name, transport_type
	# 5. Ground_fare : city_name, transport_type

# z=hotel_system()
# z.main1()









