(define (domain laundry_domain)
	(:predicates (ClothInRegion1 ?cloth)
				 (ClothInRegion2 ?cloth)
				 (ClothInRegion3 ?cloth)
				 (ClothInRegion4 ?cloth)
				 (ClothInBasket ?cloth ?basket)
				 (ClothInWasher ?cloth ?washer)
				 (BasketInNearLoc ?basket)
				 (BasketInFarLoc ?basket)
				 (BasketNearLocClear ?basket ?cloth)
				 (BasketFarLocClear ?basket ?cloth)
				 (WasherDoorOpen ?washer)
	)


	(:action load_basket_from_region_1
		:parameters (?cloth ?basket)
		:precondition (and (ClothInRegion1 ?cloth)
						   (not (BasketInFarLoc ?basket))
						   (BasketInNearLoc ?basket))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion1 ?cloth)))
	)

	(:action load_basket_from_region_2
		:parameters (?cloth ?basket)
		:precondition (and (ClothInRegion2 ?cloth)
						   (not (BasketInNearLoc ?basket))
						   (BasketInFarLoc ?basket))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion2 ?cloth)))
	)

	(:action load_basket_from_region_3
		:parameters (?cloth ?basket)
		:precondition (and (ClothInRegion3 ?cloth)
						   (not (BasketInNearLoc ?basket))
						   (BasketInFarLoc ?basket))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion3 ?cloth)))
	)

	(:action load_basket_from_region_4
		:parameters (?cloth ?basket)
		:precondition (and (ClothInRegion4 ?cloth)
						   (not (BasketInNearLoc ?basket))
						   (BasketInFarLoc ?basket))
		:effect (and (ClothInBasket ?cloth ?basket)
					 (not (ClothInRegion4 ?cloth)))
	)


	(:action clear_basket_far_loc
		:parameters (?basket ?cloth ?washer)
		:precondition (and (not (BasketFarLocClear ?basket ?cloth))
						   (WasherDoorOpen ?washer)
						   (BasketInNearLoc ?basket))
		:effect (BasketFarLocClear ?basket ?cloth)
	)

	(:action clear_basket_near_loc
		:parameters (?basket ?cloth)
		:precondition (and (not (BasketNearLocClear ?basket ?cloth))
						   (BasketInFarLoc ?basket))
		:effect (BasketNearLocClear ?basket ?cloth)
	)


	(:action move_basket_to_washer
		:parameters (?basket ?cloth)
		:precondition (and (not (BasketInNearLoc ?basket))
						   (BasketNearLocClear ?basket ?cloth))
		:effect (and (BasketInNearLoc ?basket)
		             (not (BasketInFarLoc ?basket)))
	)

	(:action move_basket_from_washer
		:parameters (?basket ?cloth)
		:precondition (and (not (BasketInFarLoc ?basket))
						   (BasketFarLocClear ?basket ?cloth))
		:effect (and (BasketInFarLoc ?basket)
		             (not (BasketInNearLoc ?basket)))
	)


	(:action open_washer
		:parameters (?washer)
		:precondition (not (WasherDoorOpen ?washer))
		:effect (WasherDoorOpen ?washer)
	)

	(:action close_washer
		:parameters (?washer)
		:precondition (WasherDoorOpen ?washer)
		:effect (not (WasherDoorOpen ?washer))
	)


	(:action load_washer_from_basket
		:parameters (?cloth ?basket ?washer)
		:precondition (and (ClothInBasket ?cloth ?basket)
					 	   (WasherDoorOpen ?washer)
					 	   (not (ClothInRegion2 ?cloth))
					 	   (not (ClothInRegion3 ?cloth))
					 	   (not (ClothInRegion4 ?cloth))
					 	   (BasketInNearLoc ?basket))
		:effect (and (ClothInWasher ?cloth ?washer)
					 (not (ClothInBasket ?cloth ?basket)))
	)

	(:action load_washer_from_region_1
		:parameters (?cloth ?basket ?washer)
		:precondition (and (ClothInRegion1 ?cloth)
					 	   (WasherDoorOpen ?washer))
		:effect (ClothInWasher ?cloth ?washer)
	)

	(:action unload_washer_into_basket
		:parameters (?cloth ?basket ?washer)
		:precondition (and (ClothInWasher ?cloth ?washer)
						   (WasherDoorOpen ?washer)
					 	   (BasketInNearLoc ?basket))
		:effect (and (not (ClothInWasher ?cloth ?washer))
					 (ClothInBasket ?cloth ?basket))
	)
)