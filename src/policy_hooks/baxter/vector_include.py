def get_vector(num_cans):
    state_vector_include = {
        'baxter': ['lArmPose', 'lGripper', 'rArmPose', 'rGripper']
    }

    for i in range(num_cans):
        state_vector_include['can{0}'.format(i)] = ['pose']

    action_vector_include = {
        'baxter': ['lArmPose', 'lGripper', 'rArmPose', 'rGripper']
    }

    target_vector_include = {
        'cloth_target_1': ['value'],
        'cloth_target_2': ['value'],
        'cloth_target_3': ['value'],
        'cloth_target_4': ['value'],
        'cloth_target_5': ['value'],
        'cloth_target_6': ['value'],
        'middle_target_1': ['value'],
        'middle_target_2': ['value'],
    }

    for i in range(num_cans):
        target_vector_include['can{0}_end_target'.format(i)] = ['value']

    return state_vector_include, action_vector_include, target_vector_include

