def get_vector(config):

    state_vector_include = {
        'pr2': ['pose', 'gripper']
    }

    for i in range(num_cans):
        state_vector_include['can{0}'.format(i)] = ['pose']

    action_vector_include = {
        'pr2': ['pose', 'gripper']
    }

    target_vector_include = {
        'middle_target': ['value'],
        'left_target_1': ['value'],
        'right_target_1': ['value'],
        'left_target_2': ['value'],
        'right_target_2': ['value'],
    }

    for i in range(num_cans):
        target_vector_include['can{0}_end_target'.format(i)] = ['value']

    return state_vector_include, action_vector_include, target_vector_include
