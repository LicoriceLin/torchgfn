from math import log,exp
mode_dict={
        0:'KREDHQN',
        1:'CGPAST',
        2:'MLVIFWY',
    }

def simple_pattern_v0(seq:str):
    s=0.
    seq_len=len(seq)
    if seq_len in [0,20]:
        return log(s+1e-5)
    elif seq_len<=4:
        s+=2*seq_len
    elif seq_len<=9:
        s+=10-2*(seq_len-5)
    elif seq_len<=14:
        s+=2*(seq_len-10)
    elif seq_len<20:
        s+=10-2*(seq_len-15)
        
    pos_score=10/len(seq)
    for i,a in enumerate(seq):
        if a in mode_dict[i%3]:
            s+=pos_score
    return log(s+1e-5)

simple_pattern=simple_pattern_v0

def simple_pattern_v1(seq:str,
        max_length:int=20,
        len_score:int=10,
        pos_score:int=10,
        beta:float=1.,
        hard_maxlen_penalty:bool=True,
        log_s:bool=False):
    '''
    rewards would be e**(v0_rewards)
    '''
    s=0.
    seq_len=len(seq)/max_length
    m= seq_len>=1 if hard_maxlen_penalty else seq_len>1
    if seq_len==0 or m:
        s=s+1e-5
        if log_s:
           s=log(s)
        return s+beta
    elif seq_len<=0.25:
        s+=len_score*4*seq_len
    elif seq_len<=0.5:
        s+=len_score*(1-4*(seq_len-0.25))
    elif seq_len<=0.75:
        s+=len_score*4*(seq_len-0.5)
    elif seq_len<=1:
        s+=len_score*(1-4*(seq_len-0.75))
    
    p=pos_score/len(seq)
    for i,a in enumerate(seq):
        if a in mode_dict[i%3]:
            s+=p
    s=s+1e-5
    if log_s:
        s=log(s)
    return s+beta