def batch_states(states, xp, phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        xp (module): numpy or cupy
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    
    states = [phi(s) for s in states]
    len_col = len(states)
    if(len_col > 0):
        len_row = len(states[0])
    else:
        len_row = 0
    result = []
    for ix in range(len_row):
        result.append(xp.asarray([row[ix] for row in states]))
    return result
