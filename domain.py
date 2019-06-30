import hotelsystem
import flightsystem
import random

# def selecting_intent(alloted_states,state):
#     m=[0,1]
#     state[0]=random.choice(m)
#     if state[0]==0:
#         # state[1]=random.choice(n)
#         main1(alloted_states)
#     else:
#         main(alloted_states)
#     print("states->>",state)

def domain_main():
    h=hotelsystem.hotel_system()
    f=flightsystem.flight_system()
    print("in domain call")
    alloted_state = [0,0,0,0,0]
    state=[0,0]
    m=[0,1]
    state[0]=random.choice(m)
    if state[0]==0:
        # state[1]=random.choice(n)
        f.main1(alloted_state)
    else:
        h.main(alloted_state)

domain_main()
