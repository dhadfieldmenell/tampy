#!/bin/bash

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.test_laundry_domain --debug

# Isolation test
# nose2 test.test_pma.test_basket_domain.TestBasketDomain.cloth_grasp_isolation --debug
#
# nose2 test.test_pma.test_basket_domain.TestBasketDomain.cloth_putdown_isolation --debug
#
# nose2 test.test_pma.test_basket_domain.TestBasketDomain.move_to_isolation --debug
#
# nose2 test.test_pma.test_basket_domain.TestBasketDomain.basket_grasp_isolation --debug
#
# nose2 test.test_pma.test_basket_domain.TestBasketDomain.basket_putdown_isolation --debug

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.moveholding_cloth_isolation --debug

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.moveholding_basket_isolation --debug

nose2 test.test_pma.test_basket_domain.TestBasketDomain.test_pose_suggester --debug


# nose2 test.test_pma.test_basket_domain.TestBasketDomain.find_cloth_position --debug

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.collision_debug_env --debug

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.laundry_basket_mesh --debug

# nose2 test.test_pma.test_basket_domain.TestBasketDomain.test_basket_domain --debug

# Collision Predicates Check
# nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_obstructs --debug

# nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_obstructs_holding --debug

# nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_r_collides --debug
#
# nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_r_collides --debug

# nose2 test.test_core.test_util_classes.test_baxter_predicates.TestBaxterPredicates.test_in_gripper_cloth --debug

# nose2 test.test_core.test_util_classes.test_baxter_sampling.TestBaxterSampling.test_resample_eereachable_ver --debug
