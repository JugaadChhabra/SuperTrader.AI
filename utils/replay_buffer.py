import numpy as np
import warnings

class ReplayBuffer:
    def __init__(self,capacity,state_shape,seed=None,prioritized=False,alpha=0.6):
        self.capacity=capacity
        self.prioritized=prioritized
        self.alpha=alpha
        self.position=0
        self.size=0
        self.rng=np.random.default_rng(seed)
        self.states=np.zeros((capacity,*state_shape),dtype=np.float32)
        self.actions=np.zeros((capacity,),dtype=np.int32)
        self.rewards=np.zeros((capacity,),dtype=np.float32)
        self.next_states=np.zeros((capacity,*state_shape),dtype=np.float32)
        self.dones=np.zeros((capacity,),dtype=np.bool_)
        self.priorities=np.ones((capacity,),dtype=np.float32) if prioritized else None

    def push(self,state,action,reward,next_state,done):
        self.states[self.position]=state
        self.actions[self.position]=action
        self.rewards[self.position]=reward
        self.next_states[self.position]=next_state
        self.dones[self.position]=done
        if self.prioritized:
            max_prio=self.priorities.max() if self.size>0 else 1.0
            self.priorities[self.position]=max_prio
        self.position=(self.position+1)%self.capacity
        self.size=min(self.size+1,self.capacity)

    def sample(self,batch_size,beta=0.4):
        if self.size==0:
            warnings.warn("Sampling from empty buffer. Returning None.")
            return None
        replace=True if self.size<batch_size else False
        if self.prioritized:
            probs=self.priorities[:self.size]**self.alpha
            probs/=probs.sum()
            indices=self.rng.choice(self.size,batch_size,p=probs,replace=replace)
            weights=(self.size*probs[indices])**(-beta)
            weights/=weights.max()
            weights=weights.astype(np.float32)
        else:
            indices=self.rng.choice(self.size,batch_size,replace=replace)
            weights=np.ones(batch_size,dtype=np.float32)
        batch=dict(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            indices=indices,
            weights=weights
        )
        return batch

    def update_priorities(self,indices,new_priorities):
        if not self.prioritized:
            return
        for idx,prio in zip(indices,new_priorities):
            self.priorities[idx]=max(prio,1e-6)

    def __len__(self):
        return self.size

