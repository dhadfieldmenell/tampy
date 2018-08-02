def get_vec(num_cans):
    state_vector_include = {
        'baxter': ['pose', 'gripper'] 
    }

    for i in range(num_cans):
        state_vector_include['can{0}'.format(i)] = ['pose']

    action_vector_include = {
        'pr2': ['pose', 'gripper']
    }

    return state_vector_include, action_vector_include

