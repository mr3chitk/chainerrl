def batch_states(states, xp, phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        xp (module): numpy or cupy
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    
    states  = [phi(s) for s in states]
    len_col = len(states)
    len_row = len(states[0])
    result  = [xp.asarray([row[ix] for row in states]) for ix in range(len_row)]
    return result
