(define (domain laundry_domain)
	(:predicates (ClothInRegion1 ?cloth)
				 (ClothInRegion2 ?cloth)
				 (ClothInRegion3 ?cloth)
				 (ClothInRegion4 ?cloth)
				 (RobotInRegion1 ?robot)
				 (RobotInRegion2 ?robot)
				 (RobotInRegion3 ?robot)
				 (RobotInRegion4 ?robot)
				 (ClothInBasket ?cloth ?basket)
				 (ClothInWasher ?cloth ?washer)
				 (BasketNearWasher ?basket ?washer)
				 (BasketNearLocClear ?basket ?cloth)
				 (BasketFarLocClear ?basket ?cloth)
				 (WasherDoorOpen ?washer)
	)


	(:action load_basket_from_region_1
		:parameters ?cloth ?basket
		:precondition (and (ClothInRegion1 ?cloth)
						   (BasketNearWasher ?basket ?washer))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion1 ?cloth)))
	)

	(:action load_basket_from_region_2
		:parameters ?cloth ?basket
		:precondition (and (ClothInRegion2 ?cloth)
						   (not (BasketNearWasher ?basket ?washer)))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion2 ?cloth)))
	)

	(:action load_basket_from_region_3
		:parameters ?cloth ?basket
		:precondition (and (ClothInRegion3 ?cloth)
						   (not (BasketNearWasher ?basket ?washer)))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion3 ?cloth)))
	)

	(:action load_basket_from_region_4
		:parameters ?cloth ?basket
		:precondition (and (ClothInRegion4 ?cloth)
						   (not (BasketNearWasher ?basket ?washer)))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion4 ?cloth)))
	)


	(:action clear_basket_far_loc
		:paramters ?basket ?cloth
		:precondition (not (BasketFarLocClear ?basket ?cloth))
		:effect (BasketFarLocClear ?basket ?cloth)
	)

	(:action clear_basket_near_loc
		:paramters ?basket ?cloth
		:precondition (not (BasketNearLocClear ?basket ?cloth))
		:effect (BasketNearLocClear ?basket ?cloth)
	)


	(:action move_basket_to_washer
		:parameters ?basket
		:precondition (and (not (BasketNearWasher ?basket ?washer))
						   (BaxterNearLocClear ?basket ?cloth))
		:effect (BasketNearWasher ?basket ?washer)
	)

	(:action move_basket_from_washer
		:parameters ?basket
		:precondition (and (BasketNearWasher ?basket ?washer)
						   (BasketFarLocClear ?basket ?washer))
		:effect (not (BasketNearWasher ?basket ?washer))
	)


	(:action open_washer
		:parameters ?basket
		:precondition (not (WasherDoorOpen ?washer))
		:effect (WasherDoorOpen ?washer)
	)

	(:action close_washer
		:parameters ?washer
		:precondition (WasherDoorOpen ?washer)
		:effect (not (WasherDoorOpen ?washer))
	)


	(:action load_washer
		:parameters ?cloth ?basket ?washer
		:precondition (and (not (ClothInWasher ?cloth ?washer)
					 	   (ClothInBasket ?cloth ?basket)
					 	   (WasherDoorOpen ?washer)
					 	   (not (ClothInRegion2 ?cloth))
					 	   (not (ClothInRegion3 ?cloth))
					 	   (not (ClothInRegion4 ?cloth)))
					 	   (BasketNearWasher ?basket ?washer))
		:effect (and (ClothInWasher ?cloth ?washer)
					 (not (WasherDoorOpen ?washer))
					 (not (ClothInBasket ?cloth ?basket)))
	)

	(:action unload_washer
		:parameters ?washer ?cloth ?table
		:precondition (and (ClothInWasher ?cloth ?washer)
						   (WasherDoorOpen ?washer)
					 	   (BasketNearWasher ?basket ?washer))
		:effect (and (not (ClothInWasher ?cloth ?washer)
					 (not (WasherDoorOpen ?washer))
					 (ClothInBasket ?cloth ?basket))
	)
)