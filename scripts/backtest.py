def cost(position , prev_position , price , bp_cost=0.0001):
    notional_amount=abs(position)
    reversed_position=prev_position*position < 0
    total_cost=notional_amount*bp_cost*(2 if reversed_position else 1)
    return total_cost
